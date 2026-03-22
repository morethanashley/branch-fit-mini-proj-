from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pickle
import numpy as np
import json
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import secrets
import random
from io import BytesIO
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from dotenv import load_dotenv
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

print("🚀 Starting BranchFit - Fixed Branch Tests...")

# Load components
try:
    # Load dataset for questions
    df = pd.read_csv('balanced_dataset_augmented.csv')
    all_questions = list(df.columns[1:])
    
    import openpyxl
    wb = openpyxl.load_workbook('branchfit_questions_final.xlsx')
    ws = wb.active
    
    branch_questions = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        question_text = row[0].strip()
        branch = row[1].strip()
        if branch not in branch_questions:
            branch_questions[branch] = []
        branch_questions[branch].append(question_text)
    
    # Create index lookup: branch -> list of question indices
    branch_question_indices = {}
    
    # Map app branch names to excel branch names if they differ
    branch_name_map = {
        'Information Technology/CSE': 'IT/CSE'
    }
    
    for branch, questions in branch_questions.items():
        # Reverse map for checking (e.g. if loop yields 'IT/CSE', we map it to 'Information Technology/CSE' for the app dictionary key)
        app_branch_name = branch
        for app_name, excel_name in branch_name_map.items():
            if excel_name == branch:
                app_branch_name = app_name
                break
                
        indices = []
        # Normalize questions from Excel for matching
        normalized_excel_qs = [' '.join(q.strip().lower().split()) for q in questions]
        for i, q in enumerate(all_questions):
            from_csv = ' '.join(q.strip().lower().split())
            if from_csv in normalized_excel_qs:
                indices.append(i)
        branch_question_indices[app_branch_name] = indices
    
    print(f"✓ Loaded branch question mapping")
    for branch, indices in branch_question_indices.items():
        print(f"  {branch}: {len(indices)} questions")
    
    # Load model and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('branch_labels.json', 'r') as f:
        branch_labels = json.load(f)
    
    print(f"✓ Loaded {len(all_questions)} questions")
    print(f"✓ Model: {model.__class__.__name__}")
    
except Exception as e:
    print(f"❌ Error loading components: {e}")
    exit(1)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# MongoDB setup
load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client['branchfit']
users_collection = db['users']
results_collection = db['test_results']

# Storage
users = {}
test_sessions = {}

# Branch information
BRANCHES = {
    'Computer Engineering': {
        'description': 'Design and develop computer hardware, embedded systems, and digital circuits',
        'skills': ['Hardware Design', 'Digital Circuits', 'Embedded Systems', 'VLSI Design'],
        'icon': 'fas fa-microchip',
        'color': 'primary'
    },
    'EXTC': {
        'description': 'Electronics and Telecommunication - signals, communication systems, and networks',
        'skills': ['Signal Processing', 'Communication Systems', 'RF Engineering', 'Network Protocols'],
        'icon': 'fas fa-broadcast-tower',
        'color': 'success'
    },
    'Electrical': {
        'description': 'Power systems, electrical machines, and energy distribution',
        'skills': ['Power Systems', 'Electrical Machines', 'Control Systems', 'Power Electronics'],
        'icon': 'fas fa-bolt',
        'color': 'warning'
    },
    'Information Technology/CSE': {
        'description': 'Software development, databases, algorithms, and information systems',
        'skills': ['Programming', 'Database Management', 'Web Development', 'Data Structures'],
        'icon': 'fas fa-laptop-code',
        'color': 'info'
    },
    'Mechanical': {
        'description': 'Mechanical systems, manufacturing, thermodynamics, and design',
        'skills': ['Mechanical Design', 'Thermodynamics', 'Manufacturing', 'CAD/CAM'],
        'icon': 'fas fa-cogs',
        'color': 'danger'
    }
}

# Pre-computed question categories for fast selection
QUESTION_CATEGORIES = {
    'foundation': [0, 1, 2, 3, 4],  # First 5 questions - always ask these
    'computer_eng': [],
    'extc': [],
    'electrical': [],
    'it_cse': [],
    'mechanical': []
}

# Categorize questions by keywords (pre-computed for speed)
def categorize_questions():
    """Pre-categorize questions by branch keywords for fast lookup."""
    keywords = {
        'computer_eng': ['hardware', 'circuits', 'electronic', 'digital', 'components'],
        'extc': ['signals', 'communication', 'transmission', 'frequency', 'networks'],
        'electrical': ['electrical', 'voltage', 'current', 'power', 'electricity'],
        'it_cse': ['software', 'code', 'database', 'programming', 'apps'],
        'mechanical': ['mechanical', 'machines', 'motion', 'forces', 'materials']
    }
    
    for i, question in enumerate(all_questions):
        question_lower = question.lower()
        max_matches = 0
        best_category = 'foundation'
        
        for category, category_keywords in keywords.items():
            matches = sum(1 for keyword in category_keywords if keyword in question_lower)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        if best_category != 'foundation' and i >= 5:  # Skip first 5 for foundation
            QUESTION_CATEGORIES[best_category].append(i)
        elif i < 5:
            QUESTION_CATEGORIES['foundation'].append(i)

# Initialize question categories
categorize_questions()

def get_fast_prediction(responses):
    """Fast prediction using the trained model."""
    features = np.array([3.0] * len(all_questions))  # Neutral default
    
    for q_idx, response in responses.items():
        if q_idx < len(features):
            features[q_idx] = float(response)
    
    features_scaled = scaler.transform(features.reshape(1, -1))
    probabilities = model.predict_proba(features_scaled)[0]
    
    branch_probs = {}
    for label_str, branch_name in branch_labels.items():
        label_idx = int(label_str)
        if label_idx < len(probabilities):
            branch_probs[branch_name] = probabilities[label_idx]
    
    return branch_probs

def get_branch_specific_score(responses, target_branch):
    """Calculate a more accurate branch-specific score."""
    if not responses:
        return 0.5  # Neutral score
    
    # Get model prediction
    model_probs = get_fast_prediction(responses)
    base_score = model_probs.get(target_branch, 0)
    
    # Calculate response-based score for the target branch
    branch_to_category = {
        'Computer Engineering': 'computer_eng',
        'EXTC': 'extc',
        'Electrical': 'electrical',
        'Information Technology/CSE': 'it_cse',
        'Mechanical': 'mechanical'
    }
    
    if target_branch not in branch_to_category:
        return base_score
    
    category = branch_to_category[target_branch]
    relevant_questions = QUESTION_CATEGORIES[category]
    
    # Calculate average response for relevant questions
    relevant_responses = [responses[q] for q in relevant_questions if q in responses]
    
    if relevant_responses:
        avg_response = sum(relevant_responses) / len(relevant_responses)
        # Convert 1-5 scale to 0-1 probability
        response_score = (avg_response - 1) / 4  # 1->0, 3->0.5, 5->1
        
        # Combine model prediction with response-based score
        combined_score = 0.6 * base_score + 0.4 * response_score
        return combined_score
    
    return base_score

def select_next_question_fast(responses, asked_questions, question_count, target_branch=None):
    """Fast adaptive question selection."""
    available = [i for i in range(len(all_questions)) if i not in asked_questions]
    
    if not available:
        return None
    
    # For branch-specific tests, focus heavily on that branch
    if target_branch:
        # Get only questions belonging to this branch
        branch_indices = branch_question_indices.get(target_branch, [])
        available_branch_qs = [
            i for i in branch_indices 
            if i not in asked_questions
        ]
        if available_branch_qs:
            return random.choice(available_branch_qs)
        else:
            return None  # All branch questions exhausted
    
    # General test logic
    # Phase 1: Foundation questions (first 5)
    if question_count < 5:
        foundation_available = [q for q in QUESTION_CATEGORIES['foundation'] if q not in asked_questions]
        if foundation_available:
            return foundation_available[0]
    
    # Phase 2: Adaptive selection based on current predictions
    if len(responses) >= 3:
        current_probs = get_fast_prediction(responses)
        
        # Get top 2 branches
        sorted_branches = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_branch = sorted_branches[0][0]
        
        # Map branch names to categories
        branch_to_category = {
            'Computer Engineering': 'computer_eng',
            'EXTC': 'extc',
            'Electrical': 'electrical',
            'Information Technology/CSE': 'it_cse',
            'Mechanical': 'mechanical'
        }
        
        # Get questions from top branch category
        if top_branch in branch_to_category:
            category = branch_to_category[top_branch]
            category_questions = [q for q in QUESTION_CATEGORIES[category] if q not in asked_questions]
            
            if category_questions:
                # Add some randomness to avoid predictable patterns
                if len(category_questions) > 3:
                    return random.choice(category_questions[:3])  # Pick from top 3
                else:
                    return category_questions[0]
    
    # Phase 3: Random selection from remaining questions
    return random.choice(available)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            user = users_collection.find_one({'username': username})
            if user and check_password_hash(user['password'], password):
                session['user'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        except (ServerSelectionTimeoutError, OperationFailure) as e:
            flash('Database is temporarily unavailable. Please try again later or contact the administrator.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        try:
            existing_user = users_collection.find_one({'username': username})
            if existing_user:
                flash('Username already exists', 'error')
            else:
                users_collection.insert_one({
                    'username': username,
                    'email': email,
                    'password': generate_password_hash(password),
                    'created_at': datetime.now().isoformat()
                })
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
        except (ServerSelectionTimeoutError, OperationFailure) as e:
            flash('Database is temporarily unavailable. Please try again later or contact the administrator.', 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/admin/login', methods=['POST'])
def admin_login():
    username = request.form.get('username')
    password = request.form.get('password')
    admin_user = os.getenv('ADMIN_USERNAME')
    admin_pass = os.getenv('ADMIN_PASSWORD')
    
    if username == admin_user and password == admin_pass:
        session['admin'] = True
        flash('Admin login successful!', 'success')
        return redirect(url_for('admin_dashboard'))
    else:
        flash('Invalid admin credentials', 'error')
        return redirect(url_for('login', tab='admin'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('login'))
    
    try:
        all_users = list(users_collection.find({}, {'_id': 0, 'password': 0}).sort('created_at', -1))
        total_users = len(all_users)
        
        all_tests = list(results_collection.find({}, {'_id': 0}).sort('timestamp', -1))
        total_tests = len(all_tests)
        
        users_with_tests = set(test.get('username') for test in all_tests if test.get('username'))
        users_tested_count = len(users_with_tests)
        
        branch_counts = {branch: 0 for branch in BRANCHES.keys()}
        total_confidence = 0
        
        recent_tests = all_tests[:10]
        
        user_performance = {}
        for test in all_tests:
            username = test.get('username')
            top_branch = test.get('top_branch')
            confidence = float(test.get('confidence', 0))
            
            if top_branch in branch_counts:
                branch_counts[top_branch] += 1
            elif top_branch:
                branch_counts[top_branch] = 1
                
            total_confidence += confidence
            
            if username and username not in user_performance:
                user_performance[username] = {
                    'username': username,
                    'test_count': 0,
                    'latest_branch': top_branch,
                    'latest_confidence': confidence
                }
            if username:
                user_performance[username]['test_count'] += 1
                
        avg_confidence = total_confidence / total_tests if total_tests > 0 else 0
        most_popular = max(branch_counts.items(), key=lambda x: x[1])[0] if total_tests > 0 else "N/A"
        
        user_table_data = []
        for u in all_users:
            uname = u.get('username')
            u_stats = user_performance.get(uname)
            user_table_data.append({
                'username': uname,
                'email': u.get('email'),
                'created_at': u.get('created_at'),
                'test_count': u_stats['test_count'] if u_stats else 0,
                'latest_branch': u_stats['latest_branch'] if u_stats else None,
                'latest_date': next((t.get('timestamp') for t in all_tests if t.get('username') == uname), None)
            })
            
        return render_template('admin_dashboard.html',
                             total_users=total_users,
                             total_tests=total_tests,
                             users_tested_count=users_tested_count,
                             avg_confidence=avg_confidence,
                             most_popular_branch=most_popular,
                             branch_counts=branch_counts,
                             recent_tests=recent_tests,
                             user_table_data=user_table_data,
                             user_performance=list(user_performance.values()))
                             
    except (ServerSelectionTimeoutError, OperationFailure) as e:
        flash('Database is temporarily unavailable.', 'error')
        return redirect(url_for('home'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Admin logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', branches=BRANCHES)

@app.route('/general-test')
def general_test():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    session_id = f"{session['user']}_{datetime.now().timestamp()}"
    
    test_sessions[session_id] = {
        'user': session['user'],
        'type': 'general',
        'responses': {},
        'asked_questions': set(),
        'question_count': 0,
        'start_time': datetime.now().isoformat()
    }
    
    session['test_session'] = session_id
    session['test_type'] = 'general'
    
    print(f"✓ Created general test session: {session_id}")
    return render_template('test_start.html', test_type='General Branch Fit Test')

@app.route('/branch-test/<path:branch>')
def branch_test(branch):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if branch not in BRANCHES:
        flash('Invalid branch selected', 'error')
        return redirect(url_for('dashboard'))
    
    session_id = f"{session['user']}_{branch}_{datetime.now().timestamp()}"
    
    test_sessions[session_id] = {
        'user': session['user'],
        'type': 'branch',
        'target_branch': branch,
        'responses': {},
        'asked_questions': set(),
        'question_count': 0,
        'start_time': datetime.now().isoformat()
    }
    
    session['test_session'] = session_id
    session['test_type'] = 'branch'
    session['target_branch'] = branch
    
    print(f"✓ Created {branch} focused test session: {session_id}")
    return render_template('test_start.html', 
                         test_type=f'{branch} Suitability Test',
                         branch_info=BRANCHES[branch])

@app.route('/question')
def question():
    if 'user' not in session or 'test_session' not in session:
        return redirect(url_for('login'))
    
    session_id = session['test_session']
    if session_id not in test_sessions:
        flash('Test session expired. Please start a new test.', 'error')
        return redirect(url_for('dashboard'))
    
    test_session = test_sessions[session_id]
    
    # Set question limits: 30 for general, 12 for branch tests
    if test_session['type'] == 'general':
        max_questions = 30
    else:
        target_b = test_session.get('target_branch', '')
        max_questions = len(branch_question_indices.get(target_b, []))
        if max_questions == 0:
            max_questions = 12  # fallback
    
    # Check if we've reached the limit
    if test_session['question_count'] >= max_questions:
        print(f"📊 Completed maximum {max_questions} questions")
        return redirect(url_for('results'))
    
    # Fast early stopping check
    if test_session['question_count'] >= 10 and len(test_session['responses']) >= 10:
        if test_session['type'] == 'general':
            current_probs = get_fast_prediction(test_session['responses'])
            max_confidence = max(current_probs.values())
            if max_confidence >= 0.80:
                print(f"🎯 Stopping early due to high confidence: {max_confidence:.1%}")
                return redirect(url_for('results'))
        else:
            # For branch tests, use branch-specific scoring
            target_branch = test_session.get('target_branch')
            if target_branch:
                branch_score = get_branch_specific_score(test_session['responses'], target_branch)
                if branch_score >= 0.80 or branch_score <= 0.20:  # Very high or very low
                    print(f"🎯 Stopping early - {target_branch} score: {branch_score:.1%}")
                    return redirect(url_for('results'))
    
    # Get next question (fast selection)
    target_branch = test_session.get('target_branch') if test_session['type'] == 'branch' else None
    question_idx = select_next_question_fast(
        test_session['responses'], 
        test_session['asked_questions'],
        test_session['question_count'],
        target_branch
    )
    
    if question_idx is None:
        print("🔚 No more questions available")
        return redirect(url_for('results'))
    
    session['current_question_idx'] = question_idx
    
    # Get question text
    question_text = all_questions[question_idx]
    
    question_data = {
        'id': question_idx,
        'question': question_text,
        'type': 'scale'
    }
    
    # Calculate progress
    progress = ((test_session['question_count'] + 1) / max_questions) * 100
    
    # Get current prediction
    current_prediction = None
    if test_session['question_count'] > 2:
        try:
            if test_session['type'] == 'branch':
                target_branch = test_session.get('target_branch')
                if target_branch:
                    branch_score = get_branch_specific_score(test_session['responses'], target_branch)
                    current_prediction = f"{target_branch} Fit: {branch_score*100:.0f}%"
            else:
                current_probs = get_fast_prediction(test_session['responses'])
                top_branch = max(current_probs.items(), key=lambda x: x[1])
                current_prediction = f"{top_branch[0]} ({top_branch[1]*100:.0f}%)"
        except:
            pass
    
    return render_template('question.html', 
                         question=question_data,
                         progress=min(progress, 100),
                         question_num=test_session['question_count'] + 1,
                         total_questions=max_questions,
                         current_prediction=current_prediction)

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    if 'user' not in session or 'test_session' not in session:
        return redirect(url_for('login'))
    
    session_id = session['test_session']
    if session_id not in test_sessions:
        flash('Test session expired. Please start a new test.', 'error')
        return redirect(url_for('dashboard'))
    
    test_session = test_sessions[session_id]
    answer = request.form.get('answer')
    
    if not answer:
        flash('Please select an answer', 'error')
        return redirect(url_for('question'))
    
    try:
        answer_value = int(answer)
        
        question_idx = session.get('current_question_idx')
        
        if question_idx is not None:
            # Record the answer
            test_session['responses'][question_idx] = answer_value
            test_session['asked_questions'].add(question_idx)
            test_session['question_count'] += 1
            
            print(f"📝 Q{test_session['question_count']}: Answer {answer_value}")
            
            # Show current prediction (only occasionally to save time)
            if test_session['question_count'] % 3 == 0:  # Every 3rd question
                if test_session['type'] == 'branch':
                    target_branch = test_session.get('target_branch')
                    if target_branch:
                        branch_score = get_branch_specific_score(test_session['responses'], target_branch)
                        print(f"🎯 {target_branch} fit: {branch_score*100:.1f}%")
                else:
                    current_probs = get_fast_prediction(test_session['responses'])
                    top_branch = max(current_probs.items(), key=lambda x: x[1])
                    print(f"🎯 Current prediction: {top_branch[0]} ({top_branch[1]*100:.1f}%)")
        
        return redirect(url_for('question'))
        
    except Exception as e:
        print(f"Error processing answer: {e}")
        flash('Error processing your answer. Please try again.', 'error')
        return redirect(url_for('question'))

@app.route('/results')
def results():
    if 'user' not in session or 'test_session' not in session:
        return redirect(url_for('login'))
    
    session_id = session['test_session']
    if session_id not in test_sessions:
        flash('Test session expired. Please start a new test.', 'error')
        return redirect(url_for('dashboard'))
    
    test_session = test_sessions[session_id]
    
    fit_label = None
    fit_color = None
    
    # Get results based on test type
    if test_session['type'] == 'branch':
        # Branch-specific test results
        target_branch = test_session.get('target_branch')
        if target_branch:
            branch_score = get_branch_specific_score(test_session['responses'], target_branch)
            
            if branch_score >= 0.70:
                fit_label = 'Strong Match'
                fit_color = 'success'
            elif branch_score >= 0.40:
                fit_label = 'Good Match'
                fit_color = 'warning'
            else:
                fit_label = 'Low Match'
                fit_color = 'danger'
            
            # Create results focused on the target branch
            branch_results = [{
                'branch': target_branch,
                'probability': branch_score * 100,
                'info': BRANCHES.get(target_branch, {})
            }]
            
            # Add other branches for comparison (using model predictions)
            model_probs = get_fast_prediction(test_session['responses'])
            for branch_name, probability in model_probs.items():
                if branch_name != target_branch:
                    branch_results.append({
                        'branch': branch_name,
                        'probability': probability * 100,
                        'info': BRANCHES.get(branch_name, {})
                    })
        else:
            # Fallback to model predictions
            current_probs = get_fast_prediction(test_session['responses'])
            branch_results = []
            for branch_name, probability in current_probs.items():
                branch_results.append({
                    'branch': branch_name,
                    'probability': probability * 100,
                    'info': BRANCHES.get(branch_name, {})
                })
    else:
        # General test results using model predictions
        current_probs = get_fast_prediction(test_session['responses'])
        branch_results = []
        for branch_name, probability in current_probs.items():
            branch_results.append({
                'branch': branch_name,
                'probability': probability * 100,
                'info': BRANCHES.get(branch_name, {})
            })
    
    # Sort by probability
    branch_results.sort(key=lambda x: x['probability'], reverse=True)
    
    print(f"🏆 Test completed! {test_session['question_count']} questions asked")
    print(f"🎯 Final prediction: {branch_results[0]['branch']} ({branch_results[0]['probability']:.1f}%)")
    
    # Save test result to MongoDB (include all branch scores for PDF)
    try:
        results_collection.insert_one({
            'username': session['user'],
            'test_type': test_session.get('type'),
            'target_branch': test_session.get('target_branch'),
            'top_branch': branch_results[0]['branch'],
            'confidence': branch_results[0]['probability'],
            'questions_asked': test_session['question_count'],
            'timestamp': datetime.now().isoformat(),
            'all_branch_scores': [{'branch': r['branch'], 'score': r['probability']} for r in branch_results]
        })
    except (ServerSelectionTimeoutError, OperationFailure):
        flash('Result could not be saved to the database. Your results are shown below.', 'error')

    return render_template('results.html', 
                         results=branch_results,
                         test_type=test_session.get('type', 'general'),
                         target_branch=test_session.get('target_branch'),
                         questions_asked=test_session['question_count'],
                         confidence=branch_results[0]['probability']/100 if branch_results else 0,
                         fit_label=fit_label,
                         fit_color=fit_color)

@app.route('/download-result')
def download_result():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        last_result = results_collection.find_one(
            {'username': session['user']},
            sort=[('timestamp', -1)],
            projection={'_id': 0}
        )
    except (ServerSelectionTimeoutError, OperationFailure):
        flash('Database is temporarily unavailable. Please try again later.', 'error')
        return redirect(url_for('dashboard'))
    if not last_result:
        flash('No test result found to download.', 'error')
        return redirect(url_for('dashboard'))

    # Use all_branch_scores if present, else build from top_branch + confidence (legacy)
    top_branch = last_result.get('top_branch', '—')
    confidence = float(last_result.get('confidence', 0))
    all_scores = last_result.get('all_branch_scores')
    if all_scores:
        all_scores = sorted(all_scores, key=lambda x: float(x.get('score', 0)), reverse=True)
    else:
        all_scores = [{'branch': top_branch, 'score': confidence}]
        for b in BRANCHES:
            if b != top_branch:
                all_scores.append({'branch': b, 'score': 0})

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    content_width = 6.5 * inch
    navy = colors.HexColor('#1a237e')
    light_grey = colors.HexColor('#f5f5f5')
    light_blue = colors.HexColor('#e3f2fd')
    green = colors.HexColor('#4caf50')
    orange = colors.HexColor('#ff9800')
    red = colors.HexColor('#f44336')

    flow = []

    # 1. HEADER: navy full-width, "BranchFit" + tagline
    header_style = ParagraphStyle(
        name='HeaderTitle', parent=styles['Normal'], fontSize=22, textColor=colors.white,
        alignment=1, spaceAfter=4, fontName='Helvetica-Bold'
    )
    header_sub_style = ParagraphStyle(
        name='HeaderSub', parent=styles['Normal'], fontSize=11, textColor=colors.white,
        alignment=1, fontName='Helvetica-Oblique'
    )
    header_table = Table([
        [Paragraph('BranchFit', header_style)],
        [Paragraph('AI Powered Branch Recommendation System', header_sub_style)]
    ], colWidths=[content_width])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), navy),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    flow.append(header_table)
    flow.append(Spacer(1, 0.4 * inch))

    # 2. STUDENT INFO: light grey box, two columns
    ts = last_result.get('timestamp', '')
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            ts = dt.strftime('%B %d, %Y at %I:%M %p')
        except Exception:
            pass
    test_type = last_result.get('test_type', 'general')
    test_type_label = 'Branch Specific' if test_type == 'branch' else 'General'
    info_data = [
        ['Student:', last_result.get('username', '—')],
        ['Date & Time:', ts or '—'],
        ['Test Type:', test_type_label],
        ['Questions Asked:', str(last_result.get('questions_asked', '—'))],
    ]
    info_table = Table(info_data, colWidths=[content_width * 0.35, content_width * 0.65])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    flow.append(info_table)
    flow.append(Spacer(1, 0.35 * inch))

    # 3. TOP RECOMMENDED BRANCH: light blue box, star + branch + confidence
    top_style = ParagraphStyle(
        name='TopBranch', parent=styles['Normal'], fontSize=16, fontName='Helvetica-Bold',
        spaceAfter=0, spaceBefore=0
    )
    top_content = Paragraph(f'&#9733; {top_branch} &nbsp;&nbsp; {confidence:.1f}%', top_style)
    top_table = Table([[top_content]], colWidths=[content_width])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_blue),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
    ]))
    flow.append(Paragraph('<b>Top Recommended Branch</b>', ParagraphStyle(name='Sec', parent=styles['Normal'], fontSize=12, spaceBefore=6, spaceAfter=8)))
    flow.append(top_table)
    flow.append(Spacer(1, 0.35 * inch))

    # 4. COMPLETE BRANCH ANALYSIS: table with Branch, Score, bar, Match level
    bar_width_pts = 120
    analysis_header = [Paragraph('<b>Branch</b>', styles['Normal']), Paragraph('<b>Score</b>', styles['Normal']),
                      Paragraph('<b>Score Bar</b>', styles['Normal']), Paragraph('<b>Match Level</b>', styles['Normal'])]
    analysis_rows = [analysis_header]
    for i, item in enumerate(all_scores):
        br = item.get('branch', '—')
        score = float(item.get('score', 0))
        score_str = f'{score:.1f}%'
        # Bar: one cell colored (width proportional), one empty
        bar_inner_width = max(0, bar_width_pts * (score / 100.0))
        if score >= 70:
            bar_color = green
            match_label, match_hex = 'Strong Match', '#4caf50'
        elif score >= 40:
            bar_color = orange
            match_label, match_hex = 'Good Match', '#ff9800'
        else:
            bar_color = red
            match_label, match_hex = 'Low Match', '#f44336'
        bar_second_width = max(1, bar_width_pts - bar_inner_width)
        bar_outer = Table([['', '']], colWidths=[bar_inner_width, bar_second_width])
        bar_outer.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), bar_color),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#eeeeee')),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        match_para = Paragraph(f'<font color="{match_hex}">{match_label}</font>',
                               ParagraphStyle(name='Match', parent=styles['Normal'], fontSize=9))
        row_bg = light_blue if br == top_branch else (colors.white if i % 2 == 0 else colors.HexColor('#fafafa'))
        analysis_rows.append([br, score_str, bar_outer, match_para])
    analysis_table = Table(analysis_rows,
                           colWidths=[content_width * 0.35, 0.9 * inch, 1.7 * inch, 1.2 * inch])
    analysis_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]
    for idx, item in enumerate(all_scores):
        r = idx + 1
        br = item.get('branch', '')
        row_bg = light_blue if br == top_branch else (colors.white if idx % 2 == 0 else colors.HexColor('#fafafa'))
        analysis_styles.append(('BACKGROUND', (0, r), (-1, r), row_bg))
    analysis_table.setStyle(TableStyle(analysis_styles))
    flow.append(Paragraph('<b>Complete Branch Analysis</b>', ParagraphStyle(name='Sec2', parent=styles['Normal'], fontSize=12, spaceBefore=10, spaceAfter=8)))
    flow.append(analysis_table)
    flow.append(Spacer(1, 0.5 * inch))

    # 5. FOOTER: line, tagline, date
    footer_line = Table([['']], colWidths=[content_width])
    footer_line.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), 1, colors.grey)]))
    flow.append(footer_line)
    flow.append(Spacer(1, 0.15 * inch))
    flow.append(Paragraph(
        'Generated by BranchFit - AI Powered Branch Recommendation System',
        ParagraphStyle(name='Footer', parent=styles['Normal'], fontSize=9, alignment=1, textColor=colors.grey)
    ))
    flow.append(Paragraph(
        datetime.now().strftime('%B %d, %Y'),
        ParagraphStyle(name='FooterDate', parent=styles['Normal'], fontSize=8, alignment=1, textColor=colors.grey)
    ))
    doc.build(flow)
    buffer.seek(0)
    filename = f"BranchFit_Result_{last_result.get('username', 'report')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

@app.route('/test-history')
def test_history():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        tests = list(results_collection.find(
            {'username': session['user']},
            {'_id': 0}
        ).sort('timestamp', -1))
    except (ServerSelectionTimeoutError, OperationFailure):
        flash('Database is temporarily unavailable. Please try again later.', 'error')
        tests = []
    return render_template('test_history.html', tests=tests)

if __name__ == '__main__':
    print("="*70)
    print("🌐 BranchFit Server: http://localhost:5000")
    print("⚡ Fixed Branch Tests - Accurate Individual Scoring")
    print("📊 General Test: 30 questions | Branch Test: 20 questions")
    print("🎯 Branch tests now focus on target branch questions")
    print("="*70)
    
    app.run(debug=True, port=5000, host='0.0.0.0')