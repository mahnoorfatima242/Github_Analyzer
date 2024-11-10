# github_analytics.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import time
from typing import List, Dict, Any
import json
from streamlit import cache_data
import logging
import sys
import numpy as np
from plotly.subplots import make_subplots
import re
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class GitHubAPI:
    def __init__(self, token: str):
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.rate_limit = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _update_rate_limit(self, response):
        self.rate_limit = int(response.headers.get('X-RateLimit-Limit', 0))
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
        
        reset_time = datetime.fromtimestamp(self.rate_limit_reset).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Rate Limits: {self.rate_limit_remaining}/{self.rate_limit} (Reset: {reset_time})")

    def _make_request(self, url: str, params: dict = None) -> dict:
        """Make an API request with improved error handling."""
        try:
            logger.info(f"Request: GET {url} {params if params else ''}")
            
            response = self.session.get(url, params=params)
            self._update_rate_limit(response)

            if response.status_code == 204:  # No content
                return {}  # Return empty dict instead of None
                
            if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                wait_time = max(self.rate_limit_reset - time.time(), 0)
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time)
                    return self._make_request(url, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            return {}  # Return empty dict on error

    def get_organization(self, org_name: str) -> dict:
        url = f"{self.base_url}/orgs/{org_name}"
        return self._make_request(url)

    def get_repositories(self, org_name: str, page: int = 1) -> list:
        url = f"{self.base_url}/orgs/{org_name}/repos"
        params = {
            'page': page,
            'per_page': 100,
            'sort': 'updated',
            'direction': 'desc'
        }
        return self._make_request(url, params)

    def get_repository_details(self, org_name: str, repo_name: str) -> dict:
        """Get repository details with additional metadata for framework detection."""
        try:
            # Get basic repo info
            url = f"{self.base_url}/repos/{org_name}/{repo_name}"
            repo_data = self._make_request(url)
            
            if not repo_data:
                return {}
                
            # Get repository contents to check for framework-related files
            contents_url = f"{self.base_url}/repos/{org_name}/{repo_name}/contents"
            contents = self._make_request(contents_url)
            
            # Look for key files that indicate framework usage
            framework_indicators = {
                'tensorflow': ['tensorflow', 'requirements.txt', 'environment.yml'],
                'pytorch': ['torch', 'requirements.txt', 'environment.yml'],
                'keras': ['keras', 'requirements.txt', 'environment.yml'],
                'scikit-learn': ['sklearn', 'requirements.txt', 'environment.yml'],
            }
            
            detected_frameworks = set()
            if isinstance(contents, list):
                for item in contents:
                    filename = item.get('name', '').lower()
                    for framework, indicators in framework_indicators.items():
                        if any(indicator in filename for indicator in indicators):
                            detected_frameworks.add(framework)
            
            # Add detected frameworks to repo data
            repo_data['detected_frameworks'] = list(detected_frameworks)
            
            # Add enhanced topic analysis
            repo_data['ai_topics'] = [
                topic for topic in repo_data.get('topics', [])
                if any(keyword in topic for keyword in [
                    'ai', 'ml', 'deep-learning', 'machine-learning', 
                    'neural', 'tensorflow', 'pytorch', 'keras'
                ])
            ]
                    
            return repo_data
            
        except Exception as e:
            logger.error(f"Error getting repository details for {org_name}/{repo_name}: {str(e)}")
            return {}

    def get_repository_languages(self, org_name: str, repo_name: str) -> dict:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/languages"
        return self._make_request(url)

    def get_contributors(self, org_name: str, repo_name: str) -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/contributors"
        params = {'per_page': 100, 'anon': 'true'}
        return self._make_request(url, params)

    def get_commit_activity(self, org_name: str, repo_name: str) -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/stats/commit_activity"
        return self._make_request(url)

    def get_code_frequency(self, org_name: str, repo_name: str) -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/stats/code_frequency"
        return self._make_request(url)

    def get_pull_requests(self, org_name: str, repo_name: str, state: str = 'all') -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/pulls"
        params = {'state': state, 'per_page': 100}
        return self._make_request(url, params)

def is_ai_repository(repo: dict) -> bool:
    ai_keywords = {
        'ai', 'machine-learning', 'deep-learning', 'neural', 
        'tensorflow', 'pytorch', 'keras', 'ml', 'artificial-intelligence',
        'data-science', 'nlp', 'computer-vision'
    }
    
    name_lower = repo['name'].lower()
    description = (repo.get('description') or '').lower()
    topics = set(repo.get('topics', []))
    
    return bool(
        any(kw in name_lower.replace('-', '') for kw in ai_keywords) or
        any(kw in description for kw in ai_keywords) or
        topics.intersection(ai_keywords)
    )

def validate_token(self):
    """Validate GitHub token by making a test API call."""
    try:
        url = f"{self.base_url}/user"
        response = self.session.get(url)
        
        if response.status_code == 401:
            raise ValueError(
                "Invalid GitHub token. Please ensure you have provided a valid token "
                "with the required permissions (repo, read:org, read:user)."
            )
        elif response.status_code != 200:
            raise ValueError(
                f"GitHub API error during token validation: {response.status_code} - {response.text}"
            )
        
        # Update rate limit info from the response
        self._update_rate_limit(response)
        return True
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Connection error during token validation: {str(e)}")

def _update_rate_limit(self, response):
        """Update rate limit information from response headers."""
        try:
            self.rate_limit = int(response.headers.get('X-RateLimit-Limit', 0))
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
            
            reset_time = datetime.fromtimestamp(self.rate_limit_reset)
            reset_time_str = reset_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add warning if rate limit is getting low
            if self.rate_limit_remaining < 10:
                logger.warning(
                    f"Rate limit running low: {self.rate_limit_remaining}/{self.rate_limit} "
                    f"(Reset at {reset_time_str})"
                )
            else:
                logger.info(
                    f"Rate Limits: {self.rate_limit_remaining}/{self.rate_limit} "
                    f"(Reset: {reset_time_str})"
                )
        except Exception as e:
            logger.error(f"Error updating rate limits: {str(e)}")

def _handle_response_error(self, response, url: str):
        """Handle different types of API errors with specific messages."""
        error_messages = {
            401: "Unauthorized: Please check your GitHub token and permissions.",
            403: "Forbidden: Rate limit exceeded or insufficient permissions.",
            404: "Not Found: The requested resource does not exist.",
            422: "Validation Failed: The request data was invalid.",
        }
        
        error_msg = error_messages.get(
            response.status_code,
            f"API Error: {response.status_code} - {response.text}"
        )
        
        # Add response details for debugging
        debug_info = {
            'status_code': response.status_code,
            'url': url,
            'response_text': response.text[:200] + '...' if len(response.text) > 200 else response.text,
            'headers': dict(response.headers)
        }
        
        logger.error(f"API Error: {error_msg}")
        logger.debug(f"Debug information: {json.dumps(debug_info, indent=2)}")
        
        return error_msg

def _make_request(self, url: str, params: dict = None) -> dict:
    if True:
        """Make an API request with improved error handling and retries."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Request: GET {url} {params if params else ''}")
                
                response = self.session.get(url, params=params, timeout=10)
                self._update_rate_limit(response)

                # Handle rate limiting
                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    wait_time = max(self.rate_limit_reset - time.time(), 0)
                    if wait_time > 0:
                        logger.warning(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
                        continue

                # Handle other error responses
                if not response.ok:
                    error_msg = self._handle_response_error(response, url)
                    if attempt < max_retries - 1:
                        retry_delay *= 2  # Exponential backoff
                        logger.warning(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return {'error': error_msg}

                return response.json()

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout for {url}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {'error': 'Request timeout'}

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {'error': f'Request failed: {str(e)}'}

        return {'error': 'Max retries exceeded'}

def analyze_repository_topics(repo: dict) -> Dict[str, Any]:
    """Analyze repository topics with improved categorization."""
    topics = repo.get('topics', [])
    
    categories = {
        'ai_ml': {
            'ai', 'machine-learning', 'deep-learning', 'neural', 
            'tensorflow', 'pytorch', 'keras', 'ml', 'artificial-intelligence',
            'data-science', 'nlp', 'computer-vision', 'reinforcement-learning',
            'neural-network', 'machine-learning-algorithms', 'deep-neural-networks',
            'ai-tools', 'ml-ops', 'ml-pipeline'
        },
        'data_science': {
            'data-science', 'data-analytics', 'big-data', 'data-visualization',
            'data-mining', 'data-analysis', 'data-processing', 'etl',
            'business-intelligence', 'statistics', 'predictive-analytics'
        },
        'natural_language': {
            'nlp', 'natural-language-processing', 'text-mining', 
            'language-model', 'text-analysis', 'speech-recognition',
            'text-generation', 'chatbot', 'translation'
        },
        'computer_vision': {
            'computer-vision', 'image-processing', 'object-detection',
            'facial-recognition', 'image-classification', 'opencv',
            'image-segmentation', 'video-processing'
        },
        'robotics_rl': {
            'robotics', 'reinforcement-learning', 'autonomous-systems',
            'robot-control', 'rl', 'deep-reinforcement-learning'
        }
    }
    
    topic_analysis = {
        'raw_topics': topics,
        'categories': defaultdict(int),
        'primary_category': None,
        'ai_subcategories': []
    }
    
    # Count matches in each category
    for topic in topics:
        for category, keywords in categories.items():
            if topic in keywords:
                topic_analysis['categories'][category] += 1
    
    # Determine primary category
    if topic_analysis['categories']:
        topic_analysis['primary_category'] = max(
            topic_analysis['categories'].items(),
            key=lambda x: x[1]
        )[0]
        
        # Get all AI-related subcategories
        topic_analysis['ai_subcategories'] = [
            category for category, count in topic_analysis['categories'].items()
            if count > 0 and category != 'data_science'  # Exclude general data science
        ]
    
    return topic_analysis



def analyze_ai_tools(repo: dict) -> Dict[str, int]:
    """Analyze AI tools and frameworks mentioned in repository with improved detection."""
    ai_tools = {
        'tensorflow': 0,
        'pytorch': 0,
        'keras': 0,
        'scikit-learn': 0,
        'hugging-face': 0,
        'opencv': 0,
        'nltk': 0,
        'spacy': 0,
        'fastai': 0,
        'mxnet': 0,
        'caffe': 0,
        'theano': 0,
        'paddle': 0,
        'transformers': 0,
        'xgboost': 0,
        'lightgbm': 0
    }
    
    # Sources to check
    sources = [
        repo.get('description', '').lower(),
        ' '.join(repo.get('topics', [])).lower(),
        repo.get('name', '').lower(),
        # Add readme content if available
        repo.get('readme_content', '').lower()
    ]
    
    # Framework variations to catch more mentions
    framework_variants = {
        'tensorflow': ['tensorflow', 'tf-', 'tf2', 'tf1'],
        'pytorch': ['pytorch', 'torch'],
        'scikit-learn': ['scikit-learn', 'sklearn'],
        'hugging-face': ['hugging-face', 'huggingface', 'transformers'],
        'opencv': ['opencv', 'cv2'],
    }
    
    # Check all sources for mentions
    for source in sources:
        for tool in ai_tools.keys():
            # Check main tool name
            if tool in source:
                ai_tools[tool] += 1
            
            # Check variations if they exist
            if tool in framework_variants:
                for variant in framework_variants[tool]:
                    if variant in source:
                        ai_tools[tool] += 1
                        break
    
    # Check requirements files for more accurate detection
    requirements = repo.get('requirements_content', '').lower()
    if requirements:
        for tool in ai_tools.keys():
            if tool in requirements:
                ai_tools[tool] += 2  # Weight requirements file mentions higher
    
    return ai_tools

def analyze_developer_activity(github_api: GitHubAPI, org_name: str, repo_name: str) -> Dict[str, Any]:
    """Analyze detailed developer activity patterns."""
    activity = {
        'commit_patterns': defaultdict(int),
        'contribution_frequency': defaultdict(int),
        'active_times': defaultdict(int),
        'top_contributors': [],
        'collaboration_score': 0,
        'code_churn': [],
        'review_stats': defaultdict(int),
        'contribution_types': defaultdict(int)
    }
    
    # Get commit frequency
    commit_data = github_api.get_commit_activity(org_name, repo_name)
    if commit_data:
        for week in commit_data:
            for day_idx, count in enumerate(week.get('days', [])):
                day_name = calendar.day_name[day_idx]
                activity['commit_patterns'][day_name] += count

    # Get code frequency (additions/deletions)
    code_freq = github_api.get_code_frequency(org_name, repo_name)
    if code_freq:
        activity['code_churn'] = [
            {
                'week': datetime.fromtimestamp(week[0]).strftime('%Y-%m-%d'),
                'additions': week[1],
                'deletions': abs(week[2])
            }
            for week in code_freq
        ]

    # Get pull requests
    pull_requests = github_api.get_pull_requests(org_name, repo_name)
    if pull_requests:
        for pr in pull_requests:
            activity['review_stats']['total_prs'] += 1
            if pr.get('merged_at'):
                activity['review_stats']['merged_prs'] += 1
            activity['review_stats']['comments'] += pr.get('comments', 0)
            activity['contribution_types']['pull_requests'] += 1

    # Get top contributors
    contributors = github_api.get_contributors(org_name, repo_name)
    if contributors:
        activity['top_contributors'] = [
            {
                'login': c['login'],
                'contributions': c['contributions'],
                'type': c.get('type', 'User'),
                'contribution_type': get_contribution_type(c['contributions'])
            }
            for c in contributors[:10]  # Top 10 contributors
        ]
        
        # Calculate collaboration score
        total_contributors = len(contributors)
        total_contributions = sum(c['contributions'] for c in contributors)
        if total_contributors > 0:
            avg_contributions = total_contributions / total_contributors
            contribution_variance = np.var([c['contributions'] for c in contributors])
            activity['collaboration_score'] = (avg_contributions / (contribution_variance + 1)) * 100

    return activity

def get_contribution_type(contributions: int) -> str:
    """Categorize contributor based on contribution count."""
    if contributions >= 100:
        return "Core Contributor"
    elif contributions >= 50:
        return "Regular Contributor"
    elif contributions >= 10:
        return "Active Contributor"
    else:
        return "Occasional Contributor"

def get_organization_data(github_api: GitHubAPI, org_name: str) -> Dict[str, Any]:
    """Collect comprehensive data for an organization with improved error handling."""
    try:
        logger.info(f"Starting analysis for organization: {org_name}")
        
        # Create a progress bar and status in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        org_data = github_api.get_organization(org_name)
        if not org_data:
            logger.error(f"Failed to fetch data for {org_name}")
            return None

        metrics = {
            'name': org_name,
            'total_repos': 0,
            'ai_repos': 0,
            'total_stars': 0,
            'total_forks': 0,
            'total_issues': 0,
            'languages': defaultdict(int),
            'contributors': set(),
            'monthly_activity': defaultdict(int),
            'repos_data': [],
            'contributor_stats': defaultdict(int),
            'ai_tools_usage': defaultdict(int),
            'technology_breakdown': defaultdict(int),
            'development_metrics': {
                'commit_frequency': defaultdict(int),
                'code_churn': [],
                'pr_stats': defaultdict(int)
            }
        }

        # Get all repositories
        repos_list = []
        page = 1
        while True:
            try:
                repos = github_api.get_repositories(org_name, page)
                if not repos or not isinstance(repos, list):
                    break
                repos_list.extend(repos)
                page += 1
                if len(repos) < 100:
                    break
            except Exception as e:
                logger.error(f"Error fetching repositories page {page}: {str(e)}")
                break

        total_repos = len(repos_list)
        logger.info(f"Found {total_repos} repositories for {org_name}")

        # Process repositories
        for idx, repo in enumerate(repos_list):
            try:
                # Update progress
                progress = (idx + 1) / total_repos
                progress_bar.progress(progress)
                status_text.text(f"Processing {repo['name']} ({idx + 1}/{total_repos})")
                
                logger.info(f"Processing repository: {repo['name']}")
                
                # Basic metrics
                repo_data = {
                    'name': repo['name'],
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'issues': repo.get('open_issues_count', 0),
                    'size': repo.get('size', 0),
                    'created_at': repo.get('created_at', ''),
                    'updated_at': repo.get('updated_at', ''),
                    'is_ai': is_ai_repository(repo),
                    'topics': repo.get('topics', []),
                    'language': repo.get('language', 'Unknown')
                }

                metrics['total_repos'] += 1
                metrics['total_stars'] += repo_data['stars']
                metrics['total_forks'] += repo_data['forks']
                metrics['total_issues'] += repo_data['issues']

                if repo_data['is_ai']:
                    metrics['ai_repos'] += 1

                # Get languages
                try:
                    languages = github_api.get_repository_languages(org_name, repo['name'])
                    if languages and isinstance(languages, dict):
                        for lang, bytes_count in languages.items():
                            metrics['languages'][lang] += bytes_count
                        repo_data['languages'] = languages
                    else:
                        repo_data['languages'] = {}
                except Exception as e:
                    logger.error(f"Error fetching languages for {repo['name']}: {str(e)}")
                    repo_data['languages'] = {}

                # Analyze topics and categories
                topic_analysis = analyze_repository_topics(repo)
                repo_data['topic_analysis'] = topic_analysis
                for category, count in topic_analysis['categories'].items():
                    metrics['technology_breakdown'][category] += count

                # Analyze AI tools
                ai_tools = analyze_ai_tools(repo)
                repo_data['ai_tools'] = ai_tools
                for tool, count in ai_tools.items():
                    metrics['ai_tools_usage'][tool] += count

                # Get developer activity
                activity_data = analyze_developer_activity(github_api, org_name, repo['name'])
                if activity_data:
                    repo_data.update({
                        'commit_patterns': activity_data.get('commit_patterns', {}),
                        'top_contributors': activity_data.get('top_contributors', []),
                        'collaboration_score': activity_data.get('collaboration_score', 0),
                        'code_churn': activity_data.get('code_churn', []),
                        'review_stats': activity_data.get('review_stats', {})
                    })

                    # Update monthly activity
                    for week in repo_data.get('code_churn', []):
                        if isinstance(week, dict) and 'week' in week:
                            month = week['week'][:7]  # YYYY-MM
                            metrics['monthly_activity'][month] += (
                                week.get('additions', 0) + week.get('deletions', 0)
                            )

                    # Track contributors
                    for contributor in activity_data.get('top_contributors', []):
                        if isinstance(contributor, dict) and 'login' in contributor:
                            metrics['contributors'].add(contributor['login'])
                            metrics['contributor_stats'][contributor.get('contribution_type', 'Unknown')] += 1

                metrics['repos_data'].append(repo_data)

            except Exception as e:
                logger.error(f"Error processing repository {repo.get('name', 'unknown')}: {str(e)}")
                continue

        metrics['total_contributors'] = len(metrics['contributors'])
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        logger.info(f"Analysis complete for {org_name}")
        return metrics

    except Exception as e:
        logger.error(f"Error processing {org_name}: {str(e)}")
        return None



def create_enhanced_visualizations(companies_data: List[Dict[str, Any]]):
    """Create all visualizations for the dashboard."""
    # Cache report generation
    @cache_data
    def get_cached_reports(data):
        detailed_report = generate_detailed_report(data)
        ai_report = generate_ai_report(data)
        return detailed_report, ai_report    
    # 1. Overview Dashboard
    st.header("Organization Overview")
    
    # Create metrics for each company
    cols = st.columns(len(companies_data))
    for idx, company in enumerate(companies_data):
        with cols[idx]:
            st.metric(
                label=f"{company['name']} Overview",
                value=f"{company['total_repos']} Repos",
                delta=f"{company['ai_repos']} AI Projects"
            )
            st.metric(
                label="Community Size",
                value=f"{company['total_contributors']:,} Contributors",
                delta=f"{company['total_stars']:,} Stars"
            )

    # 2. Repository Analysis
    st.header("Repository Analysis")
    
    tabs = st.tabs(["Repository Distribution", "Technology Stack", "Development Activity", "AI Focus"])
    
    with tabs[0]:
        # Repository Distribution
        fig_repos = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Repository Type Distribution",
                "Repository Growth Over Time",
                "Repository Size Distribution",
                "Stars vs Forks Analysis"
            )
        )
        
        # Repository Type Distribution
        repo_types = {
            company['name']: {
                'AI/ML': company['ai_repos'],
                'Web Dev': company['technology_breakdown'].get('web_dev', 0),
                'Mobile': company['technology_breakdown'].get('mobile', 0),
                'DevOps': company['technology_breakdown'].get('devops', 0),
                'Other': company['total_repos'] - company['ai_repos'] - 
                        sum(company['technology_breakdown'].get(k, 0) 
                            for k in ['web_dev', 'mobile', 'devops'])
            }
            for company in companies_data
        }
        
        x_data = list(repo_types.keys())
        for category in ['AI/ML', 'Web Dev', 'Mobile', 'DevOps', 'Other']:
            fig_repos.add_trace(
                go.Bar(
                    name=category,
                    x=x_data,
                    y=[repo_types[company][category] for company in x_data],
                    text=[repo_types[company][category] for company in x_data],
                    textposition='auto',
                ),
                row=1, col=1
            )

        # Repository Growth
        for company in companies_data:
            dates = sorted(company['monthly_activity'].keys())
            cumulative_repos = np.cumsum([company['monthly_activity'][date] for date in dates])
            fig_repos.add_trace(
                go.Scatter(
                    name=company['name'],
                    x=dates,
                    y=cumulative_repos,
                    mode='lines+markers'
                ),
                row=1, col=2
            )

        # Repository Size Distribution
        for company in companies_data:
            sizes = [repo['size'] for repo in company['repos_data']]
            fig_repos.add_trace(
                go.Box(
                    name=company['name'],
                    y=sizes,
                    boxpoints='outliers'
                ),
                row=2, col=1
            )

        # Stars vs Forks
        for company in companies_data:
            fig_repos.add_trace(
                go.Scatter(
                    name=company['name'],
                    x=[repo['stars'] for repo in company['repos_data']],
                    y=[repo['forks'] for repo in company['repos_data']],
                    mode='markers',
                    text=[repo['name'] for repo in company['repos_data']],
                    hovertemplate=
                    "<b>%{text}</b><br>" +
                    "Stars: %{x}<br>" +
                    "Forks: %{y}<br>",
                ),
                row=2, col=2
            )

        fig_repos.update_layout(height=800, showlegend=True, barmode='stack')
        st.plotly_chart(fig_repos, use_container_width=True)

    with tabs[1]:
        # Technology Stack Analysis
        fig_tech = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Programming Language Distribution",
                "Technology Category Breakdown",
                "Framework Usage Comparison",
                "Technology Trends"
            )
        )

        # Programming Languages
        for company in companies_data:
            languages = dict(sorted(
                company['languages'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            fig_tech.add_trace(
                go.Bar(
                    name=company['name'],
                    x=list(languages.keys()),
                    y=list(languages.values()),
                    text=list(languages.values()),
                    textposition='auto',
                ),
                row=1, col=1
            )

        # Technology Categories
        tech_categories = defaultdict(lambda: defaultdict(int))
        for company in companies_data:
            for repo in company['repos_data']:
                for category, count in repo['topic_analysis']['categories'].items():
                    tech_categories[company['name']][category] += count

        categories = set()
        for company_cats in tech_categories.values():
            categories.update(company_cats.keys())

        for category in sorted(categories):
            fig_tech.add_trace(
                go.Bar(
                    name=category,
                    x=list(tech_categories.keys()),
                    y=[tech_categories[company][category] for company in tech_categories],
                ),
                row=1, col=2
            )

        # Framework Usage
        framework_usage = defaultdict(lambda: defaultdict(int))
        for company in companies_data:
            for repo in company['repos_data']:
                for framework, count in repo.get('ai_tools', {}).items():
                    if count > 0:
                        framework_usage[company['name']][framework] += 1

        for framework in set().union(*(d.keys() for d in framework_usage.values())):
            fig_tech.add_trace(
                go.Bar(
                    name=framework,
                    x=list(framework_usage.keys()),
                    y=[framework_usage[company][framework] for company in framework_usage],
                ),
                row=2, col=1
            )

        # Technology Trends
        for company in companies_data:
            dates = sorted(company['monthly_activity'].keys())
            ai_activity = [
                sum(1 for repo in company['repos_data']
                    if repo['is_ai'] and date in repo['created_at'])
                for date in dates
            ]
            fig_tech.add_trace(
                go.Scatter(
                    name=f"{company['name']} AI Trend",
                    x=dates,
                    y=ai_activity,
                    mode='lines+markers'
                ),
                row=2, col=2
            )

        fig_tech.update_layout(height=800, showlegend=True, barmode='group')
        st.plotly_chart(fig_tech, use_container_width=True)

    with tabs[2]:
        # Development Activity Analysis
        st.subheader("Development Activity Patterns")
        
        fig_activity = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Commit Patterns by Day",
                "Contributor Distribution",
                "Code Churn Overview",
                "Pull Request Activity"
            )
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Continuing with Development Activity Analysis
        
        # Commit Patterns by Day
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for company in companies_data:
            commit_patterns = defaultdict(int)
            for repo in company['repos_data']:
                for day, count in repo.get('commit_patterns', {}).items():
                    commit_patterns[day] += count
            
            fig_activity.add_trace(
                go.Scatter(
                    name=company['name'],
                    x=days_order,
                    y=[commit_patterns[day] for day in days_order],
                    mode='lines+markers'
                ),
                row=1, col=1
            )

        # Contributor Distribution
        for company in companies_data:
            contributor_types = defaultdict(int)
            for repo in company['repos_data']:
                for contributor in repo.get('top_contributors', []):
                    contributor_types[contributor['contribution_type']] += 1
            
            fig_activity.add_trace(
                go.Bar(
                    name=company['name'],
                    x=list(contributor_types.keys()),
                    y=list(contributor_types.values()),
                    text=list(contributor_types.values()),
                    textposition='auto'
                ),
                row=1, col=2
            )

        # Code Churn Overview
        for company in companies_data:
            code_churn_data = []
            for repo in company['repos_data']:
                code_churn_data.extend(repo.get('code_churn', []))
            
            if code_churn_data:
                df = pd.DataFrame(code_churn_data)
                df['week'] = pd.to_datetime(df['week'])
                monthly_churn = df.groupby(df['week'].dt.to_period('M')).agg({
                    'additions': 'sum',
                    'deletions': 'sum'
                }).reset_index()
                
                fig_activity.add_trace(
                    go.Bar(
                        name=f"{company['name']} Additions",
                        x=monthly_churn['week'].astype(str),
                        y=monthly_churn['additions'],
                        marker_color='green'
                    ),
                    row=2, col=1
                )
                fig_activity.add_trace(
                    go.Bar(
                        name=f"{company['name']} Deletions",
                        x=monthly_churn['week'].astype(str),
                        y=-monthly_churn['deletions'],
                        marker_color='red'
                    ),
                    row=2, col=1
                )

        # Pull Request Activity
        for company in companies_data:
            pr_stats = defaultdict(int)
            for repo in company['repos_data']:
                review_stats = repo.get('review_stats', {})
                pr_stats['total'] += review_stats.get('total_prs', 0)
                pr_stats['merged'] += review_stats.get('merged_prs', 0)
                pr_stats['comments'] += review_stats.get('comments', 0)
            
            fig_activity.add_trace(
                go.Bar(
                    name=company['name'],
                    x=['Total PRs', 'Merged PRs', 'Comments'],
                    y=[pr_stats['total'], pr_stats['merged'], pr_stats['comments']],
                    text=[pr_stats['total'], pr_stats['merged'], pr_stats['comments']],
                    textposition='auto'
                ),
                row=2, col=2
            )

        fig_activity.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig_activity, use_container_width=True)

    with tabs[3]:
        # AI Focus Analysis
        st.subheader("AI Development Analysis")
        
        fig_ai = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "AI Framework Usage",
                "AI Project Growth",
                "AI Project Types",
                "AI Developer Distribution"
            )
        )

        # AI Framework Usage
        for company in companies_data:
            framework_counts = defaultdict(int)
            for repo in company['repos_data']:
                if repo['is_ai']:
                    for framework, count in repo.get('ai_tools', {}).items():
                        framework_counts[framework] += count
            
            fig_ai.add_trace(
                go.Bar(
                    name=company['name'],
                    x=list(framework_counts.keys()),
                    y=list(framework_counts.values()),
                    text=list(framework_counts.values()),
                    textposition='auto'
                ),
                row=1, col=1
            )

        # AI Project Growth
        for company in companies_data:
            ai_repos = [repo for repo in company['repos_data'] if repo['is_ai']]
            dates = pd.to_datetime([repo['created_at'] for repo in ai_repos])
            monthly_counts = pd.Series(1, index=dates).resample('M').sum().fillna(0)
            cumulative_counts = monthly_counts.cumsum()
            
            fig_ai.add_trace(
                go.Scatter(
                    name=company['name'],
                    x=cumulative_counts.index,
                    y=cumulative_counts.values,
                    mode='lines+markers'
                ),
                row=1, col=2
            )

        # AI Project Types
        ai_categories = defaultdict(lambda: defaultdict(int))
        for company in companies_data:
            for repo in company['repos_data']:
                if repo['is_ai']:
                    for category in repo['topic_analysis']['categories']:
                        if category.startswith('ai_'):
                            ai_categories[company['name']][category] += 1

        for category in set().union(*(d.keys() for d in ai_categories.values())):
            fig_ai.add_trace(
                go.Bar(
                    name=category,
                    x=list(ai_categories.keys()),
                    y=[ai_categories[company][category] for company in ai_categories],
                ),
                row=2, col=1
            )

        # AI Developer Distribution
        for company in companies_data:
            ai_contributors = defaultdict(int)
            for repo in company['repos_data']:
                if repo['is_ai']:
                    for contributor in repo.get('top_contributors', []):
                        ai_contributors[contributor['contribution_type']] += 1
            
            fig_ai.add_trace(
                go.Bar(
                    name=company['name'],
                    x=list(ai_contributors.keys()),
                    y=list(ai_contributors.values()),
                    text=list(ai_contributors.values()),
                    textposition='auto'
                ),
                row=2, col=2
            )

        fig_ai.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig_ai, use_container_width=True)

    # Additional Insights
        st.header("Key Insights")
    
    for company in companies_data:
        with st.expander(f"{company['name']} Insights"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Repository Statistics")
                st.write(f"- Total Repositories: {company['total_repos']}")
                ai_repo_percentage = (company['ai_repos']/company['total_repos']*100) if company['total_repos'] > 0 else 0
                st.write(f"- AI/ML Repositories: {company['ai_repos']} ({ai_repo_percentage:.1f}%)")
                st.write(f"- Total Stars: {company['total_stars']:,}")
                st.write(f"- Total Contributors: {company['total_contributors']:,}")
            
            with col2:
                st.markdown("### Development Metrics")
                total_commits = sum(company['monthly_activity'].values())
                
                # Calculate average collaboration score with error handling
                valid_scores = [
                    repo.get('collaboration_score', 0) 
                    for repo in company['repos_data']
                    if isinstance(repo.get('collaboration_score'), (int, float))
                ]
                avg_collab_score = np.mean(valid_scores) if valid_scores else 0
                
                st.write(f"- Total Commits: {total_commits:,}")
                st.write(f"- Average Collaboration Score: {avg_collab_score:.1f}")
                
                # Handle empty commit patterns
                try:
                    # Check if there's data and commit_patterns exists
                    if (company['repos_data'] and 
                        isinstance(company['repos_data'][0], dict) and 
                        company['repos_data'][0].get('commit_patterns')):
                        
                        commit_patterns = company['repos_data'][0]['commit_patterns']
                        if commit_patterns:
                            most_active_day = max(commit_patterns.items(), key=lambda x: x[1])[0]
                            st.write(f"- Most Active Day: {most_active_day}")
                        else:
                            st.write("- Most Active Day: No data available")
                    else:
                        st.write("- Most Active Day: No data available")
                except (IndexError, KeyError, ValueError) as e:
                    logger.warning(f"Error calculating most active day for {company['name']}: {str(e)}")
                    st.write("- Most Active Day: No data available")
                    
                    
    # Download Reports
    detailed_report_data, ai_report_data = get_cached_reports(companies_data)
    
    # Add download section
    st.markdown("---")
    
    # Export options with direct download buttons
    with st.expander("📥 Export Options"):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📊 Download Full Report",
                data=detailed_report_data,
                file_name="github_analytics_report.csv",
                mime="text/csv",
                key="download_full_report"
            )
        
        with col2:
            st.download_button(
                label="🤖 Download AI Report",
                data=ai_report_data,
                file_name="github_ai_report.csv",
                mime="text/csv",
                key="download_ai_report"
            )

def generate_detailed_report(companies_data: List[Dict[str, Any]]) -> str:
    """Generate detailed CSV report of all metrics."""
    report_data = []
    
    for company in companies_data:
        # Calculate aggregate metrics with error handling
        valid_scores = [
            repo.get('collaboration_score', 0) 
            for repo in company['repos_data']
            if isinstance(repo.get('collaboration_score'), (int, float))
        ]
        avg_collab_score = np.mean(valid_scores) if valid_scores else 0
        
        total_commits = sum(company['monthly_activity'].values())
        
        # Get top languages
        top_languages = sorted(
            company['languages'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Get most used AI frameworks
        ai_frameworks = defaultdict(int)
        for repo in company['repos_data']:
            if repo['is_ai']:
                for framework, count in repo.get('ai_tools', {}).items():
                    ai_frameworks[framework] += count
        
        top_frameworks = sorted(
            ai_frameworks.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Get most active day with error handling
        try:
            if (company['repos_data'] and 
                isinstance(company['repos_data'][0], dict) and 
                company['repos_data'][0].get('commit_patterns')):
                
                commit_patterns = company['repos_data'][0]['commit_patterns']
                most_active_day = max(commit_patterns.items(), key=lambda x: x[1])[0] if commit_patterns else 'N/A'
            else:
                most_active_day = 'N/A'
        except (IndexError, KeyError, ValueError):
            most_active_day = 'N/A'
        
        report_data.append({
            'Company': company['name'],
            'Total Repositories': company['total_repos'],
            'AI Repositories': company['ai_repos'],
            'AI Repository Percentage': f"{(company['ai_repos']/company['total_repos']*100) if company['total_repos'] > 0 else 0:.1f}%",
            'Total Stars': company['total_stars'],
            'Total Forks': company['total_forks'],
            'Total Contributors': company['total_contributors'],
            'Total Commits': total_commits,
            'Average Collaboration Score': f"{avg_collab_score:.1f}",
            'Top Languages': ', '.join(f"{lang}({count})" for lang, count in top_languages),
            'Top AI Frameworks': ', '.join(f"{framework}({count})" for framework, count in top_frameworks),
            'Most Active Day': most_active_day,
            'Core Contributors': company['contributor_stats'].get('Core Contributor', 0),
            'Regular Contributors': company['contributor_stats'].get('Regular Contributor', 0),
        })
    
    return pd.DataFrame(report_data).to_csv(index=False)



def generate_ai_report(companies_data: List[Dict[str, Any]]) -> str:
    """Generate AI-focused CSV report."""
    ai_report_data = []
    
    for company in companies_data:
        ai_repos = [repo for repo in company['repos_data'] if repo['is_ai']]
        
        # Calculate AI-specific metrics
        total_ai_stars = sum(repo['stars'] for repo in ai_repos)
        total_ai_forks = sum(repo['forks'] for repo in ai_repos)
        ai_contributors = set()
        
        for repo in ai_repos:
            for contributor in repo.get('top_contributors', []):
                ai_contributors.add(contributor['login'])
        
        # Analyze AI frameworks and tools
        framework_usage = defaultdict(int)
        for repo in ai_repos:
            for framework, count in repo.get('ai_tools', {}).items():
                framework_usage[framework] += count
        
        ai_report_data.append({
            'Company': company['name'],
            'Total AI Repositories': len(ai_repos),
            'AI Repository Percentage': f"{(len(ai_repos)/company['total_repos']*100):.1f}%",
            'AI Stars': total_ai_stars,
            'AI Forks': total_ai_forks,
            'AI Contributors': len(ai_contributors),
            'Framework Distribution': ', '.join(
                f"{framework}({count})" 
                for framework, count in sorted(
                    framework_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ),
            'Most Popular AI Repo': max(
                ai_repos,
                key=lambda x: x['stars']
            )['name'] if ai_repos else 'N/A',
            'Average AI Repo Size': f"{np.mean([repo['size'] for repo in ai_repos]):.0f} KB" if ai_repos else 'N/A',
            'Latest AI Repo': max(
                ai_repos,
                key=lambda x: x['created_at']
            )['name'] if ai_repos else 'N/A'
        })
    
    return pd.DataFrame(ai_report_data).to_csv(index=False)


def parse_company_urls(file_content: str) -> List[str]:
    """
    Parse company names or GitHub URLs from uploaded file content.
    
    Args:
        file_content: Raw content from uploaded file
        
    Returns:
        List of company/organization names cleaned and formatted
    """
    companies = []
    try:
        # Decode if bytes
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = file_content

        # Process each line
        for line in content.split('\n'):
            # Clean the line
            line = line.strip()
            
            if line:  # Skip empty lines
                # Handle different URL formats
                if 'github.com' in line:
                    # Extract org name from URL
                    # Handle both https://github.com/org and github.com/org formats
                    parts = line.split('github.com/')
                    if len(parts) > 1:
                        org_name = parts[1].strip('/')
                        # Remove any additional path components
                        org_name = org_name.split('/')[0]
                else:
                    # Assume it's just an organization name
                    org_name = line

                # Clean the organization name
                org_name = org_name.strip()
                if org_name:
                    # Remove any trailing .git if present
                    org_name = org_name.rstrip('.git')
                    companies.append(org_name)

        # Remove duplicates while preserving order
        seen = set()
        companies = [x for x in companies if x not in seen and not seen.add(x)]

        logger.info(f"Successfully parsed {len(companies)} company names")
        return companies

    except Exception as e:
        logger.error(f"Error parsing company URLs: {str(e)}")
        return []

def validate_company_name(name: str) -> bool:
    """
    Validate if a company name follows GitHub's organization naming rules.
    
    Args:
        name: Company/organization name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # GitHub organization naming rules
    # - Can contain alphanumeric characters or hyphens
    # - Cannot start with a hyphen
    # - Maximum length is 39 characters
    
    if not name:
        return False
    
    if len(name) > 39:
        return False
        
    if name.startswith('-'):
        return False
        
    # Check if contains only allowed characters
    return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*$', name))


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="GitHub Organization Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            height: 3em;
            margin: 0.5em 0;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #00ff00;
        }
        .success-message {
            color: #0f5132;
            background-color: #d1e7dd;
            border-color: #badbcc;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .error-message {
            color: #842029;
            background-color: #f8d7da;
            border-color: #f5c2c7;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.title("📊 GitHub Organization Analytics")
    st.markdown("""
        Analyze and compare GitHub organizations' development activities, 
        AI initiatives, and developer contributions.
    """)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # GitHub token input
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        help="Enter your GitHub personal access token"
    )
    
    # File upload for company list
    uploaded_file = st.sidebar.file_uploader(
        "Company List",
        type="txt",
        help="Upload a text file with one company name per line"
    )

    # Analysis options
    st.sidebar.header("Analysis Settings")
    
    # Analysis period selector
    analysis_period = st.sidebar.slider(
        "Analysis Period (months)",
        min_value=1,
        max_value=24,
        value=12,
        help="Number of months of historical data to analyze"
    )
    
    # Fork inclusion option
    include_forks = st.sidebar.checkbox(
        "Include Forks",
        value=False,
        help="Include forked repositories in the analysis"
    )

    # Reset button
    if st.sidebar.button("Reset Analysis", key="reset_analysis"):
        st.session_state.clear()
        st.rerun()

    # Show analysis status if complete
    if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
        st.sidebar.markdown(
            '<div class="success-message">✅ Analysis Complete</div>', 
            unsafe_allow_html=True
        )

    if github_token and uploaded_file:
        try:
            # Initialize or get analysis state
            if 'analysis_complete' not in st.session_state:
                st.session_state.analysis_complete = False
                st.session_state.companies_data = None

            # Perform analysis if needed
            if not st.session_state.analysis_complete:
                # Initialize GitHub API client
                github_api = GitHubAPI(github_token)
                
                # Parse company names
                companies = parse_company_urls(uploaded_file.read())
                
                if not companies:
                    st.error("No valid company names found in the uploaded file.")
                    return

                # Display initial message
                st.info("Starting analysis... This may take a few minutes.")
                
                try:
                    # Perform the analysis
                    companies_data = perform_github_analysis(github_api, companies)
                    
                    if companies_data:
                        # Store results and rerun
                        st.session_state.companies_data = companies_data
                        st.session_state.analysis_complete = True
                        st.rerun()
                    else:
                        st.error("No valid data could be retrieved for the specified companies.")
                        if 'analysis_complete' in st.session_state:
                            del st.session_state.analysis_complete
                        return
                        
                except Exception as analysis_error:
                    st.error(f"Analysis failed: {str(analysis_error)}")
                    logger.error(f"Analysis error: {str(analysis_error)}")
                    if 'analysis_complete' in st.session_state:
                        del st.session_state.analysis_complete
                    return

            # Create visualizations using cached data
            if st.session_state.companies_data:
                create_enhanced_visualizations(st.session_state.companies_data)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: Could not connect to GitHub API. {str(e)}")
            logger.error(f"Network error: {str(e)}")
            if 'analysis_complete' in st.session_state:
                del st.session_state.analysis_complete
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main execution: {str(e)}", exc_info=True)
            if 'analysis_complete' in st.session_state:
                del st.session_state.analysis_complete
    
    else:
        st.info("Please provide your GitHub token and upload a company list to begin analysis.")
        
        with st.expander("ℹ️ How to Get Started"):
            st.markdown("""
                ### Setting Up Your Analysis
                
                1. **GitHub Token**
                   - Go to GitHub → Settings → Developer settings
                   - Navigate to Personal access tokens → Tokens (classic)
                   - Generate new token with these permissions:
                     - `repo` (Full repository access)
                     - `read:org` (Read organization data)
                     - `read:user` (Read user data)
                
                2. **Company List**
                   - Create a text file (.txt)
                   - Add one company per line
                   - Use either:
                     - Organization names (e.g., "microsoft")
                     - Full GitHub URLs (e.g., "https://github.com/google")
                
                3. **Example Format**:
                ```
                microsoft
                google
                facebook
                https://github.com/netflix
                ```
                
                4. **Analysis Options**
                   - Adjust the analysis period to focus on specific timeframes
                   - Choose whether to include forked repositories
                   - Select specific metrics to analyze
            """)

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>
        GitHub Organization Analytics Tool v1.0<br>
        Created with Streamlit | Data provided by GitHub API
        </small>
        </div>
        """, 
        unsafe_allow_html=True
    )



@cache_data
def perform_github_analysis(_github_api, companies):
    """Cached function to perform GitHub analysis."""
    companies_data = []
    
    # Create progress bar and status text outside the loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_companies = len(companies)
        for idx, company in enumerate(companies):
            # Update progress
            progress = (idx + 1) / total_companies
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {company}... ({idx + 1}/{total_companies})")
            
            company_data = get_organization_data(_github_api, company)
            if company_data:
                companies_data.append(company_data)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return companies_data
        
    except Exception as e:
        # Clean up progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        logger.error(f"Error in analysis: {str(e)}")
        raise e
    
    

if __name__ == "__main__":
    main()
