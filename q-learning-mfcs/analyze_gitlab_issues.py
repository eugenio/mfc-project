#!/usr/bin/env python3
"""
GitLab Issue Analysis Script for MFC Q-Learning Project.
Retrieves, analyzes, and prioritizes GitLab issues.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the tests directory to the path to access gitlab_issue_manager
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

try:
    import gitlab
    from gitlab_issue_manager import GitLabIssueManager, IssueSeverity, IssueUrgency, IssueType
    GITLAB_AVAILABLE = True
except ImportError as e:
    GITLAB_AVAILABLE = False
    print(f"GitLab dependencies not available: {e}")
    print("Install with: pip install python-gitlab")
    sys.exit(1)


class GitLabIssueAnalyzer:
    """Analyze and prioritize GitLab issues."""
    
    def __init__(self):
        """Initialize the issue analyzer."""
        self.issue_manager = GitLabIssueManager()
        self.gl = self.issue_manager.gl
        self.project = self.issue_manager.project
        
    def get_all_issues(self, state='opened') -> List[Dict[str, Any]]:
        """
        Retrieve all issues from GitLab.
        
        Args:
            state: Issue state ('opened', 'closed', 'all')
            
        Returns:
            List of issue dictionaries
        """
        try:
            issues = self.project.issues.list(state=state, all=True)
            
            issue_data = []
            for issue in issues:
                issue_data.append({
                    'id': issue.id,
                    'iid': issue.iid,
                    'title': issue.title,
                    'description': issue.description,
                    'state': issue.state,
                    'labels': issue.labels,
                    'created_at': issue.created_at,
                    'updated_at': issue.updated_at,
                    'author': issue.author,
                    'assignees': getattr(issue, 'assignees', []),
                    'milestone': getattr(issue, 'milestone', None),
                    'web_url': issue.web_url,
                    'upvotes': getattr(issue, 'upvotes', 0),
                    'downvotes': getattr(issue, 'downvotes', 0),
                    'user_notes_count': getattr(issue, 'user_notes_count', 0)
                })
            
            return issue_data
            
        except Exception as e:
            print(f"Error retrieving issues: {e}")
            return []
    
    def analyze_issue_severity(self, issue: Dict[str, Any]) -> IssueSeverity:
        """
        Analyze issue severity based on content and labels.
        
        Args:
            issue: Issue dictionary
            
        Returns:
            Determined severity level
        """
        title = issue['title'].lower()
        description = (issue['description'] or '').lower()
        labels = [label.lower() for label in issue['labels']]
        
        # Check for explicit severity labels
        if 'critical' in labels or 'severity::critical' in labels:
            return IssueSeverity.CRITICAL
        if 'high' in labels or 'severity::high' in labels:
            return IssueSeverity.HIGH
        if 'medium' in labels or 'severity::medium' in labels:
            return IssueSeverity.MEDIUM
        if 'low' in labels or 'severity::low' in labels:
            return IssueSeverity.LOW
        
        # Analyze based on keywords
        critical_keywords = [
            'crash', 'fatal', 'corruption', 'security', 'data loss',
            'memory leak', 'deadlock', 'infinite loop'
        ]
        
        high_keywords = [
            'error', 'exception', 'fail', 'broken', 'not working',
            'performance', 'slow', 'timeout', 'regression'
        ]
        
        medium_keywords = [
            'improvement', 'enhancement', 'optimize', 'refactor',
            'feature request', 'usability'
        ]
        
        for keyword in critical_keywords:
            if keyword in title or keyword in description:
                return IssueSeverity.CRITICAL
        
        for keyword in high_keywords:
            if keyword in title or keyword in description:
                return IssueSeverity.HIGH
        
        for keyword in medium_keywords:
            if keyword in title or keyword in description:
                return IssueSeverity.MEDIUM
        
        return IssueSeverity.LOW
    
    def analyze_issue_urgency(self, issue: Dict[str, Any]) -> IssueUrgency:
        """
        Analyze issue urgency based on age, activity, and content.
        
        Args:
            issue: Issue dictionary
            
        Returns:
            Determined urgency level
        """
        labels = [label.lower() for label in issue['labels']]
        
        # Check for explicit urgency labels
        if 'urgent' in labels or 'urgency::urgent' in labels:
            return IssueUrgency.URGENT
        if 'high' in labels or 'urgency::high' in labels:
            return IssueUrgency.HIGH
        if 'medium' in labels or 'urgency::medium' in labels:
            return IssueUrgency.MEDIUM
        if 'low' in labels or 'urgency::low' in labels:
            return IssueUrgency.LOW
        
        # Calculate based on age and activity
        created_date = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
        age_days = (datetime.now().astimezone() - created_date).days
        
        # Recent issues with high activity are more urgent
        if age_days <= 7 and issue['user_notes_count'] > 3:
            return IssueUrgency.URGENT
        elif age_days <= 30 and issue['user_notes_count'] > 1:
            return IssueUrgency.HIGH
        elif age_days <= 90:
            return IssueUrgency.MEDIUM
        else:
            return IssueUrgency.LOW
    
    def analyze_issue_type(self, issue: Dict[str, Any]) -> IssueType:
        """
        Determine issue type based on labels and content.
        
        Args:
            issue: Issue dictionary
            
        Returns:
            Determined issue type
        """
        labels = [label.lower() for label in issue['labels']]
        title = issue['title'].lower()
        description = (issue['description'] or '').lower()
        
        # Check explicit labels
        label_mapping = {
            'bug': IssueType.BUG,
            'enhancement': IssueType.ENHANCEMENT,
            'feature': IssueType.ENHANCEMENT,
            'performance': IssueType.PERFORMANCE,
            'security': IssueType.SECURITY,
            'documentation': IssueType.DOCUMENTATION,
            'test': IssueType.TEST,
            'testing': IssueType.TEST
        }
        
        for label in labels:
            if label in label_mapping:
                return label_mapping[label]
        
        # Analyze based on content
        if any(word in title or word in description for word in 
               ['error', 'fail', 'broken', 'crash', 'bug', 'issue', 'problem']):
            return IssueType.BUG
        
        if any(word in title or word in description for word in 
               ['performance', 'slow', 'optimize', 'speed', 'memory', 'cpu']):
            return IssueType.PERFORMANCE
        
        if any(word in title or word in description for word in 
               ['security', 'vulnerability', 'exploit', 'attack']):
            return IssueType.SECURITY
        
        if any(word in title or word in description for word in 
               ['documentation', 'docs', 'readme', 'guide', 'tutorial']):
            return IssueType.DOCUMENTATION
        
        if any(word in title or word in description for word in 
               ['test', 'testing', 'unittest', 'pytest', 'coverage']):
            return IssueType.TEST
        
        # Default to enhancement for feature requests
        return IssueType.ENHANCEMENT
    
    def calculate_priority_score(self, severity: IssueSeverity, urgency: IssueUrgency, 
                                issue_type: IssueType, issue: Dict[str, Any]) -> int:
        """
        Calculate a priority score for the issue.
        
        Args:
            severity: Issue severity
            urgency: Issue urgency  
            issue_type: Issue type
            issue: Issue data
            
        Returns:
            Priority score (higher = more important)
        """
        # Base scores
        severity_scores = {
            IssueSeverity.CRITICAL: 40,
            IssueSeverity.HIGH: 30,
            IssueSeverity.MEDIUM: 20,
            IssueSeverity.LOW: 10
        }
        
        urgency_scores = {
            IssueUrgency.URGENT: 30,
            IssueUrgency.HIGH: 20,
            IssueUrgency.MEDIUM: 10,
            IssueUrgency.LOW: 5
        }
        
        type_scores = {
            IssueType.SECURITY: 20,
            IssueType.BUG: 15,
            IssueType.PERFORMANCE: 10,
            IssueType.TEST: 8,
            IssueType.ENHANCEMENT: 5,
            IssueType.DOCUMENTATION: 3
        }
        
        base_score = (severity_scores[severity] + 
                     urgency_scores[urgency] + 
                     type_scores[issue_type])
        
        # Additional factors
        activity_bonus = min(issue['user_notes_count'] * 2, 10)
        upvote_bonus = min(issue['upvotes'] * 3, 15)
        
        return base_score + activity_bonus + upvote_bonus
    
    def analyze_and_prioritize_issues(self) -> List[Dict[str, Any]]:
        """
        Analyze all issues and return them prioritized.
        
        Returns:
            List of analyzed and prioritized issues
        """
        print("🔍 Retrieving GitLab issues...")
        issues = self.get_all_issues()
        
        if not issues:
            print("❌ No issues found or unable to access GitLab")
            return []
        
        print(f"📊 Analyzing {len(issues)} issues...")
        
        analyzed_issues = []
        
        for issue in issues:
            severity = self.analyze_issue_severity(issue)
            urgency = self.analyze_issue_urgency(issue)
            issue_type = self.analyze_issue_type(issue)
            priority_score = self.calculate_priority_score(severity, urgency, issue_type, issue)
            
            analyzed_issue = {
                **issue,
                'analyzed_severity': severity.value,
                'analyzed_urgency': urgency.value,
                'analyzed_type': issue_type.value,
                'priority_score': priority_score,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            analyzed_issues.append(analyzed_issue)
        
        # Sort by priority score (descending)
        analyzed_issues.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return analyzed_issues
    
    def generate_priority_report(self, analyzed_issues: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive priority report.
        
        Args:
            analyzed_issues: List of analyzed issues
            
        Returns:
            Formatted report string
        """
        if not analyzed_issues:
            return "No issues to analyze."
        
        report = []
        report.append("=" * 80)
        report.append("🎯 GITLAB ISSUES ANALYSIS & PRIORITIZATION REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Total Issues Analyzed: {len(analyzed_issues)}")
        report.append("")
        
        # Summary by severity
        severity_counts = {}
        urgency_counts = {}
        type_counts = {}
        
        for issue in analyzed_issues:
            severity = issue['analyzed_severity']
            urgency = issue['analyzed_urgency']
            issue_type = issue['analyzed_type']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        report.append("📈 SUMMARY BY SEVERITY:")
        for severity, count in sorted(severity_counts.items()):
            report.append(f"   {severity.upper():>8}: {count}")
        
        report.append("")
        report.append("⏰ SUMMARY BY URGENCY:")
        for urgency, count in sorted(urgency_counts.items()):
            report.append(f"   {urgency.upper():>8}: {count}")
        
        report.append("")
        report.append("🏷️  SUMMARY BY TYPE:")
        for issue_type, count in sorted(type_counts.items()):
            report.append(f"   {issue_type.upper():>12}: {count}")
        
        report.append("")
        report.append("🔥 TOP PRIORITY ISSUES:")
        report.append("=" * 50)
        
        # Show top 10 highest priority issues
        top_issues = analyzed_issues[:10]
        
        for i, issue in enumerate(top_issues, 1):
            report.append(f"#{i} [Score: {issue['priority_score']}] Issue #{issue['iid']}")
            report.append(f"    Title: {issue['title']}")
            report.append(f"    Type: {issue['analyzed_type']} | "
                         f"Severity: {issue['analyzed_severity']} | "
                         f"Urgency: {issue['analyzed_urgency']}")
            report.append(f"    Created: {issue['created_at'][:10]}")
            report.append(f"    URL: {issue['web_url']}")
            report.append("")
        
        # Critical issues section
        critical_issues = [i for i in analyzed_issues if i['analyzed_severity'] == 'critical']
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            report.append("=" * 50)
            for issue in critical_issues:
                report.append(f"Issue #{issue['iid']}: {issue['title']}")
                report.append(f"    Priority Score: {issue['priority_score']}")
                report.append(f"    URL: {issue['web_url']}")
                report.append("")
        
        # Recommendations section
        report.append("💡 RECOMMENDATIONS:")
        report.append("=" * 50)
        
        if critical_issues:
            report.append(f"1. 🚨 Address {len(critical_issues)} critical issues immediately")
        
        high_severity = [i for i in analyzed_issues if i['analyzed_severity'] == 'high']
        if high_severity:
            report.append(f"2. ⚡ Focus on {len(high_severity)} high-severity issues next")
        
        urgent_issues = [i for i in analyzed_issues if i['analyzed_urgency'] == 'urgent']
        if urgent_issues:
            report.append(f"3. ⏱️  Handle {len(urgent_issues)} urgent issues within this week")
        
        bugs = [i for i in analyzed_issues if i['analyzed_type'] == 'bug']
        if bugs:
            report.append(f"4. 🐛 Prioritize fixing {len(bugs)} bugs for stability")
        
        security_issues = [i for i in analyzed_issues if i['analyzed_type'] == 'security']
        if security_issues:
            report.append(f"5. 🔒 Review {len(security_issues)} security issues immediately")
        
        return "\n".join(report)


def main():
    """Main function to run the analysis."""
    if not GITLAB_AVAILABLE:
        return
    
    try:
        analyzer = GitLabIssueAnalyzer()
        
        # Analyze and prioritize issues
        analyzed_issues = analyzer.analyze_and_prioritize_issues()
        
        if not analyzed_issues:
            print("No issues found to analyze.")
            return
        
        # Generate and display report
        report = analyzer.generate_priority_report(analyzed_issues)
        print(report)
        
        # Save detailed results to file
        output_file = f"gitlab_issues_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(analyzed_issues, f, indent=2, default=str)
        
        print(f"\n💾 Detailed analysis saved to: {output_file}")
        
        return analyzed_issues
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None


if __name__ == "__main__":
    main()