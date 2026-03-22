#!/usr/bin/env python3
"""
True Adaptive Questioning System V2
- Adapts after every single question
- Dynamically selects most informative questions
- Uses uncertainty sampling and information gain
- Completely avoids repetition
"""

import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score
import pandas as pd

class TrueAdaptiveSystem:
    def __init__(self):
        """Initialize the adaptive system."""
        self.load_model_and_data()
        self.reset_session()
    
    def load_model_and_data(self):
        """Load model, scaler, and question data."""
        # Load model
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load branch labels
        with open('branch_labels.json', 'r') as f:
            self.branch_labels = json.load(f)
        
        # Load all questions# Load dataset
        df = pd.read_csv('balanced_dataset_augmented.csv')
        self.all_questions = list(df.columns[1:])  # All 60 questions, skip target column
        
        print(f"✓ Loaded {len(self.all_questions)} questions")
        print(f"✓ Model expects {self.model.n_features_in_} features")
    
    def reset_session(self):
        """Reset session for new user."""
        self.responses = {}  # question_index -> response (1-5)
        self.asked_questions = set()  # Set of asked question indices
        self.question_count = 0
    
    def get_current_probabilities(self):
        """Get current branch probabilities based on answered questions."""
        # Create feature vector with neutral values (3)
        features = np.array([3.0] * len(self.all_questions))
        
        # Fill in actual responses
        for q_idx, response in self.responses.items():
            if q_idx < len(features):
                features[q_idx] = float(response)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Map to branch names
        branch_probs = {}
        for label_str, branch_name in self.branch_labels.items():
            label_idx = int(label_str)
            if label_idx < len(probabilities):
                branch_probs[branch_name] = probabilities[label_idx]
        
        return branch_probs
    
    def calculate_information_gain(self, question_idx):
        """
        Calculate expected information gain for asking a specific question.
        This measures how much the question would reduce uncertainty.
        """
        if question_idx in self.asked_questions:
            return -1  # Already asked
        
        current_probs = self.get_current_probabilities()
        current_entropy = self.calculate_entropy(list(current_probs.values()))
        
        # Simulate answering this question with different responses
        expected_entropy = 0
        response_weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Weight for responses 1,2,3,4,5
        
        for response in [1, 2, 3, 4, 5]:
            # Temporarily add this response
            temp_responses = self.responses.copy()
            temp_responses[question_idx] = response
            
            # Calculate probabilities with this response
            features = np.array([3.0] * len(self.all_questions))
            for q_idx, resp in temp_responses.items():
                if q_idx < len(features):
                    features[q_idx] = float(resp)
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            entropy = self.calculate_entropy(probabilities)
            expected_entropy += response_weights[response - 1] * entropy
        
        information_gain = current_entropy - expected_entropy
        return information_gain
    
    def calculate_entropy(self, probabilities):
        """Calculate entropy of probability distribution."""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -np.sum(probabilities * np.log2(probabilities))
    
    def calculate_uncertainty_sampling(self, question_idx):
        """
        Calculate uncertainty sampling score.
        Higher score = more uncertainty = more informative question.
        """
        if question_idx in self.asked_questions:
            return -1
        
        # Get current probabilities
        current_probs = list(self.get_current_probabilities().values())
        
        # Calculate uncertainty metrics
        max_prob = max(current_probs)
        entropy = self.calculate_entropy(current_probs)
        
        # Combine metrics (lower max_prob and higher entropy = more uncertainty)
        uncertainty_score = entropy * (1 - max_prob)
        
        return uncertainty_score
    
    def get_branch_specific_questions(self):
        """
        Identify which questions are most relevant for each branch.
        This helps focus on questions that distinguish between top candidates.
        """
        current_probs = self.get_current_probabilities()
        
        # Get top 2 branches
        sorted_branches = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_branches = [branch for branch, _ in sorted_branches[:2]]
        
        # Define branch-specific keywords (enhanced)
        branch_keywords = {
            "Computer Engineering": [
                "hardware", "circuits", "electronic", "systems", "components", 
                "voltage", "current", "electrical", "power"
            ],
            "EXTC": [
                "signals", "communication", "transmission", "networks", "frequency", 
                "spectrum", "noise", "bandwidth", "encoded", "decoded"
            ],
            "Electrical": [
                "electrical", "voltage", "current", "power", "faults", "electricity",
                "generated", "transmitted", "distributed", "protection", "load"
            ],
            "Information Technology/CSE": [
                "software", "code", "database", "algorithms", "apps", "data",
                "computer", "digital", "programming", "systems", "security"
            ],
            "Mechanical": [
                "mechanical", "machines", "motion", "forces", "stress", "vibrations", 
                "physical", "materials", "pressure", "energy", "failure"
            ]
        }
        
        # Score questions based on relevance to top branches
        question_scores = {}
        
        for i, question in enumerate(self.all_questions):
            if i in self.asked_questions:
                continue
            
            question_lower = question.lower()
            score = 0
            
            # Higher score for questions relevant to top branches
            for branch in top_branches:
                if branch in branch_keywords:
                    for keyword in branch_keywords[branch]:
                        if keyword in question_lower:
                            score += current_probs[branch]  # Weight by branch probability
            
            question_scores[i] = score
        
        return question_scores
    
    def select_next_question(self):
        """
        Intelligently select the next most informative question.
        Combines multiple strategies for optimal question selection.
        """
        if len(self.asked_questions) >= len(self.all_questions):
            return None  # All questions asked
        
        available_questions = [i for i in range(len(self.all_questions)) 
                             if i not in self.asked_questions]
        
        if not available_questions:
            return None
        
        # Strategy 1: Information Gain (most important)
        info_gains = {}
        for q_idx in available_questions:
            info_gains[q_idx] = self.calculate_information_gain(q_idx)
        
        # Strategy 2: Branch-specific relevance
        branch_scores = self.get_branch_specific_questions()
        
        # Strategy 3: Uncertainty sampling
        uncertainty_scores = {}
        for q_idx in available_questions:
            uncertainty_scores[q_idx] = self.calculate_uncertainty_sampling(q_idx)
        
        # Combine strategies with weights
        final_scores = {}
        for q_idx in available_questions:
            info_gain = info_gains.get(q_idx, 0)
            branch_score = branch_scores.get(q_idx, 0)
            uncertainty = uncertainty_scores.get(q_idx, 0)
            
            # Weighted combination
            final_score = (
                0.5 * info_gain +           # 50% information gain
                0.3 * branch_score +        # 30% branch relevance  
                0.2 * uncertainty           # 20% uncertainty
            )
            
            final_scores[q_idx] = final_score
        
        # Select question with highest score
        best_question = max(final_scores.items(), key=lambda x: x[1])
        
        return best_question[0]
    
    def answer_question(self, question_idx, response):
        """Record user's response to a question."""
        self.responses[question_idx] = response
        self.asked_questions.add(question_idx)
        self.question_count += 1
    
    def get_recommendations(self, top_n=2):
        """Get top N branch recommendations."""
        probs = self.get_current_probabilities()
        sorted_branches = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for branch, prob in sorted_branches[:top_n]:
            recommendations.append((branch, prob * 100))
        
        return recommendations
    
    def should_stop_early(self, confidence_threshold=0.8):
        """
        Determine if we can stop early due to high confidence.
        """
        probs = self.get_current_probabilities()
        max_prob = max(probs.values())
        
        # Stop if we're very confident and have asked at least 5 questions
        return max_prob >= confidence_threshold and self.question_count >= 5
    
    def get_session_summary(self):
        """Get summary of current session."""
        probs = self.get_current_probabilities()
        recommendations = self.get_recommendations()
        
        return {
            'questions_asked': self.question_count,
            'current_probabilities': probs,
            'top_recommendations': recommendations,
            'confidence': max(probs.values()),
            'can_stop_early': self.should_stop_early()
        }

def test_adaptive_system():
    """Test the adaptive system with simulated responses."""
    
    print("="*60)
    print("TESTING TRUE ADAPTIVE SYSTEM")
    print("="*60)
    
    system = TrueAdaptiveSystem()
    
    # Simulate a user interested in Computer Engineering
    # Responses that would indicate Computer Engineering preference
    ce_responses = [4, 5, 3, 5, 4]  # High for hardware/circuits questions
    
    for i in range(10):  # Ask 10 questions
        # Get next question
        q_idx = system.select_next_question()
        if q_idx is None:
            break
        
        question = system.all_questions[q_idx]
        print(f"\nQ{i+1}: {question[:80]}...")
        
        # Simulate response (in real app, this comes from user)
        if i < len(ce_responses):
            response = ce_responses[i]
        else:
            # Generate response based on question content
            if any(word in question.lower() for word in ['hardware', 'circuits', 'electronic']):
                response = 5  # Strong agreement for CE-related questions
            elif any(word in question.lower() for word in ['software', 'code', 'database']):
                response = 2  # Disagreement for IT-related questions
            else:
                response = 3  # Neutral for others
        
        print(f"Response: {response}")
        
        # Record response
        system.answer_question(q_idx, response)
        
        # Show current state
        summary = system.get_session_summary()
        print(f"Current top prediction: {summary['top_recommendations'][0][0]} ({summary['top_recommendations'][0][1]:.1f}%)")
        
        # Check if we can stop early
        if summary['can_stop_early']:
            print(f"🎯 High confidence reached! Can stop early.")
            break
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_summary = system.get_session_summary()
    print(f"Questions asked: {final_summary['questions_asked']}")
    print(f"Final confidence: {final_summary['confidence']:.1%}")
    
    print(f"\nTop recommendations:")
    for i, (branch, confidence) in enumerate(final_summary['top_recommendations']):
        print(f"  {i+1}. {branch}: {confidence:.1f}%")

if __name__ == "__main__":
    test_adaptive_system()