#!/usr/bin/env python3
"""
Automatic GitLab issue creation system

This module provides automatic issue creation when bugs or features are described.
It integrates with the main GitLab integration to provide seamless issue management.
"""

import re
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .gitlab_integration import gitlab_manager, create_bug_issue, create_enhancement_issue

class AutoIssueDetector:
    """
    Automatically detects bug reports and feature requests from text descriptions.
    Creates appropriate GitLab issues with structured information.
    """
    
    def __init__(self):
        self.bug_keywords = [
            'bug', 'error', 'issue', 'problem', 'broken', 'not working',
            'crash', 'freeze', 'hang', 'fail', 'exception', 'stack trace',
            'incorrect', 'wrong', 'unexpected', 'malfunction'
        ]
        
        self.feature_keywords = [
            'feature', 'enhancement', 'improve', 'add', 'implement',
            'suggestion', 'request', 'would like', 'could we', 'new',
            'better', 'upgrade', 'extend', 'modify', 'change'
        ]
        
        self.priority_keywords = {
            'high': ['urgent', 'critical', 'important', 'asap', 'priority', 'blocking'],
            'low': ['minor', 'nice to have', 'when possible', 'low priority', 'someday']
        }
    
    def analyze_description(self, description: str) -> Dict[str, Any]:
        """
        Analyze a text description to determine if it's a bug or feature request.
        
        Args:
            description: Text description from user
        
        Returns:
            Dictionary with analysis results
        """
        description_lower = description.lower()
        
        # Count keyword matches
        bug_score = sum(1 for keyword in self.bug_keywords if keyword in description_lower)
        feature_score = sum(1 for keyword in self.feature_keywords if keyword in description_lower)
        
        # Determine type
        if bug_score > feature_score:
            issue_type = 'bug'
            confidence = bug_score / (bug_score + feature_score) if (bug_score + feature_score) > 0 else 0
        elif feature_score > bug_score:
            issue_type = 'enhancement'
            confidence = feature_score / (bug_score + feature_score) if (bug_score + feature_score) > 0 else 0
        else:
            issue_type = 'unclear'
            confidence = 0.5
        
        # Determine priority
        priority = 'medium'  # default
        for priority_level, keywords in self.priority_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                priority = priority_level
                break
        
        # Extract potential title (first sentence or up to first period/newline)
        title_match = re.match(r'^([^.\n]+)', description.strip())
        suggested_title = title_match.group(1).strip() if title_match else "User reported issue"
        
        # Limit title length
        if len(suggested_title) > 80:
            suggested_title = suggested_title[:77] + "..."
        
        return {
            'type': issue_type,
            'confidence': confidence,
            'priority': priority,
            'suggested_title': suggested_title,
            'bug_score': bug_score,
            'feature_score': feature_score,
            'description': description
        }
    
    def create_issue_from_description(self, description: str, 
                                    force_type: Optional[str] = None) -> Optional[int]:
        """
        Automatically create a GitLab issue from a description.
        
        Args:
            description: User's description of the issue/feature
            force_type: Force issue type ('bug' or 'enhancement')
        
        Returns:
            Issue ID if created successfully, None otherwise
        """
        analysis = self.analyze_description(description)
        
        # Override type if forced
        if force_type:
            analysis['type'] = force_type
        
        # Don't create if unclear and no force
        if analysis['type'] == 'unclear' and not force_type:
            print(f"⚠️  Unclear issue type (confidence: {analysis['confidence']:.2f})")
            print("Please specify if this is a 'bug' or 'enhancement'")
            return None
        
        print(f"🔍 Detected {analysis['type']} with {analysis['confidence']:.2f} confidence")
        print(f"📋 Priority: {analysis['priority']}")
        print(f"📝 Title: {analysis['suggested_title']}")
        
        # Create appropriate issue
        if analysis['type'] == 'bug':
            return self._create_bug_from_analysis(analysis)
        elif analysis['type'] == 'enhancement':
            return self._create_enhancement_from_analysis(analysis)
        
        return None
    
    def _create_bug_from_analysis(self, analysis: Dict[str, Any]) -> Optional[int]:
        """Create a bug issue from analysis results."""
        
        # Parse description for structured information
        description = analysis['description']
        
        # Try to extract steps to reproduce
        steps_match = re.search(r'(?:steps?|reproduce|how to):(.*?)(?:\n\n|\n[A-Z]|$)', 
                               description, re.IGNORECASE | re.DOTALL)
        steps = steps_match.group(1).strip() if steps_match else None
        
        # Try to extract expected behavior
        expected_match = re.search(r'(?:expected|should):(.*?)(?:\n\n|\n[A-Z]|$)', 
                                  description, re.IGNORECASE | re.DOTALL)
        expected = expected_match.group(1).strip() if expected_match else None
        
        # Environment info
        environment = f"Automatically detected via Claude Code Assistant\nTimestamp: {datetime.now().isoformat()}"
        
        return create_bug_issue(
            title=analysis['suggested_title'],
            description=analysis['description'],
            steps_to_reproduce=steps,
            expected_behavior=expected,
            environment=environment
        )
    
    def _create_enhancement_from_analysis(self, analysis: Dict[str, Any]) -> Optional[int]:
        """Create an enhancement issue from analysis results."""
        
        # Try to extract todo items from description
        todo_patterns = [
            r'(?:todo|tasks?|steps?):\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:implement|add|create):\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:\d+\.\s+.*?)(?:\n|$)'  # Numbered lists
        ]
        
        todo_list = []
        for pattern in todo_patterns:
            matches = re.findall(pattern, analysis['description'], re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by lines and clean up
                lines = [line.strip() for line in match.split('\n') if line.strip()]
                todo_list.extend(lines)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_todos = []
        for item in todo_list:
            if item not in seen:
                seen.add(item)
                unique_todos.append(item)
        
        return create_enhancement_issue(
            title=analysis['suggested_title'],
            description=analysis['description'],
            todo_list=unique_todos if unique_todos else None,
            priority=analysis['priority']
        )

# Global instance
auto_detector = AutoIssueDetector()

def auto_create_issue(description: str, force_type: Optional[str] = None) -> Optional[int]:
    """
    Convenience function to automatically create an issue from description.
    
    Args:
        description: User's description
        force_type: Force issue type ('bug' or 'enhancement')
    
    Returns:
        Issue ID if created, None otherwise
    """
    return auto_detector.create_issue_from_description(description, force_type)

def analyze_user_input(description: str) -> Dict[str, Any]:
    """
    Convenience function to analyze user input without creating an issue.
    
    Args:
        description: User's description
    
    Returns:
        Analysis results
    """
    return auto_detector.analyze_description(description)

if __name__ == "__main__":
    # Test the auto-detection system
    test_descriptions = [
        "The GUI is not updating with real time graphs and the GPU is still active after simulation ended",
        "I want to add a model for the cathode, research online literature for platinum base and biological based cathode models",
        "There's a critical bug in the simulation that causes crashes when using large electrode areas",
        "Could we add a feature to export simulation results to different formats?"
    ]
    
    print("🧪 Testing Auto Issue Detection")
    print("=" * 50)
    
    for i, desc in enumerate(test_descriptions, 1):
        print(f"\n{i}. Testing: {desc[:60]}...")
        analysis = analyze_user_input(desc)
        print(f"   Type: {analysis['type']} (confidence: {analysis['confidence']:.2f})")
        print(f"   Priority: {analysis['priority']}")
        print(f"   Title: {analysis['suggested_title']}")
    
    print("\n✅ Auto detection test completed")