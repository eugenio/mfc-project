#!/usr/bin/env python3
import sys
sys.path.append('q-learning-mfcs/src')
from utils.gitlab_integration import gitlab_manager

if gitlab_manager.project:
    # Check for automated issues
    issues = gitlab_manager.project.issues.list(state='opened', labels=['automated'], per_page=10, order_by='created_at', sort='desc')
    print(f'Found {len(issues)} automated issues:')
    for issue in issues:
        print(f'#{issue.iid}: {issue.title} (created: {issue.created_at})')
    
    # Also check for recent issues without automated label
    all_recent = gitlab_manager.project.issues.list(state='opened', per_page=5, order_by='created_at', sort='desc')
    print(f'\nMost recent 5 open issues:')
    for issue in all_recent:
        print(f'#{issue.iid}: {issue.title} (labels: {issue.labels})')
else:
    print('No GitLab project found')