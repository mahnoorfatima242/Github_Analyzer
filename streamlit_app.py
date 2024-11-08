# github_analytics.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import time
from typing import List, Dict, Any
import json
import logging
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
        
        # Update sidebar rate limit info
        st.sidebar.metric(
            "API Rate Limit",
            f"{self.rate_limit_remaining}/{self.rate_limit}",
            f"Resets at {reset_time}"
        )

    def _make_request(self, url: str, params: dict = None) -> dict:
        try:
            logger.info(f"Request: GET {url} {params if params else ''}")
            
            response = self.session.get(url, params=params)
            self._update_rate_limit(response)

            if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                wait_time = max(self.rate_limit_reset - time.time(), 0)
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time)
                    return self._make_request(url, params)

            response.raise_for_status()
            logger.info(f"Response: {response.status_code} ({len(response.text)} bytes)")
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            return None

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

    def get_repository_languages(self, org_name: str, repo_name: str) -> dict:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/languages"
        return self._make_request(url)

    def get_contributors(self, org_name: str, repo_name: str) -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/contributors"
        params = {'per_page': 100}
        return self._make_request(url, params)

    def get_commit_activity(self, org_name: str, repo_name: str) -> list:
        url = f"{self.base_url}/repos/{org_name}/{repo_name}/stats/commit_activity"
        return self._make_request(url)

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

def parse_company_urls(file_content: str) -> List[str]:
    companies = []
    for line in file_content.decode('utf-8').split('\n'):
        line = line.strip()
        if line:
            if 'github.com' in line:
                org_name = line.split('github.com/')[-1].strip('/')
            else:
                org_name = line
            companies.append(org_name)
    return companies

def get_organization_data(github_api: GitHubAPI, org_name: str) -> Dict[str, Any]:
    try:
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        
        logger.info(f"Starting analysis for organization: {org_name}")
        status_text.text(f"Fetching organization data: {org_name}")
        
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
            'repos_data': []
        }

        # Get all repositories first
        repos_list = []
        page = 1
        status_text.text(f"Fetching repositories for {org_name}")
        
        while True:
            repos = github_api.get_repositories(org_name, page)
            if not repos:
                break
            repos_list.extend(repos)
            page += 1
            if len(repos) < 100:
                break

        total_repos = len(repos_list)
        logger.info(f"Found {total_repos} repositories for {org_name}")

        # Process repositories
        for idx, repo in enumerate(repos_list, 1):
            progress = idx / total_repos
            progress_bar.progress(progress)
            status_text.text(f"Processing {repo['name']} ({idx}/{total_repos})")
            
            logger.info(f"Processing repository: {repo['name']}")
            
            metrics['total_repos'] += 1
            metrics['total_stars'] += repo['stargazers_count']
            metrics['total_forks'] += repo['forks_count']
            metrics['total_issues'] += repo['open_issues_count']

            if is_ai_repository(repo):
                metrics['ai_repos'] += 1
                logger.info(f"AI Repository detected: {repo['name']}")

            languages = github_api.get_repository_languages(org_name, repo['name'])
            if languages:
                for lang, bytes_count in languages.items():
                    metrics['languages'][lang] += bytes_count

            contributors = github_api.get_contributors(org_name, repo['name'])
            if contributors:
                for contributor in contributors:
                    metrics['contributors'].add(contributor['login'])

            commit_activity = github_api.get_commit_activity(org_name, repo['name'])
            if commit_activity:
                for week in commit_activity:
                    date = datetime.fromtimestamp(week['week'])
                    month = date.strftime('%Y-%m')
                    metrics['monthly_activity'][month] += week['total']

        metrics['total_contributors'] = len(metrics['contributors'])
        
        progress_container.empty()
        status_text.empty()

        logger.info(f"""
Analysis Complete for {org_name}:
- Total Repositories: {metrics['total_repos']}
- AI Repositories: {metrics['ai_repos']}
- Total Stars: {metrics['total_stars']}
- Total Contributors: {metrics['total_contributors']}
""")
        
        return metrics

    except Exception as e:
        logger.error(f"Error processing {org_name}: {str(e)}")
        return None

def create_visualizations(companies_data: List[Dict[str, Any]]):
    logger.info("Creating visualizations...")

    st.header("Company Overview")
    cols = st.columns(len(companies_data))
    for idx, company in enumerate(companies_data):
        with cols[idx]:
            ai_percentage = (company['ai_repos'] / company['total_repos'] * 100) if company['total_repos'] > 0 else 0
            st.metric(
                label=company['name'],
                value=f"{company['total_repos']} Repos",
                delta=f"{ai_percentage:.1f}% AI"
            )
            st.metric(
                label="Activity",
                value=f"{company['total_stars']} Stars",
                delta=f"{company['total_contributors']} Contributors"
            )

    # Repository Distribution
    st.header("Repository Analysis")
    repo_df = pd.DataFrame([{
        'Company': company['name'],
        'AI Repositories': company['ai_repos'],
        'Other Repositories': company['total_repos'] - company['ai_repos']
    } for company in companies_data])

    fig_repos = px.bar(
        repo_df,
        x='Company',
        y=['AI Repositories', 'Other Repositories'],
        title='Repository Distribution',
        barmode='stack'
    )
    st.plotly_chart(fig_repos, use_container_width=True)

    # Technology Distribution
    col1, col2 = st.columns(2)

    with col1:
        all_languages = defaultdict(int)
        for company in companies_data:
            for lang, count in company['languages'].items():
                all_languages[lang] += count

        languages_df = pd.DataFrame([
            {'Language': lang, 'Count': count}
            for lang, count in all_languages.items()
        ]).nlargest(10, 'Count')

        fig_languages = px.pie(
            languages_df,
            values='Count',
            names='Language',
            title='Top Programming Languages'
        )
        st.plotly_chart(fig_languages, use_container_width=True)

    with col2:
        fig_ai = px.pie(
            repo_df,
            values='AI Repositories',
            names='Company',
            title='AI Development Distribution'
        )
        st.plotly_chart(fig_ai, use_container_width=True)

    # Activity Timeline
    st.header("Development Timeline")
    activity_data = []
    for company in companies_data:
        for month, count in company['monthly_activity'].items():
            activity_data.append({
                'Month': month,
                'Company': company['name'],
                'Commits': count
            })

    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        activity_df['Month'] = pd.to_datetime(activity_df['Month'])
        fig_activity = px.line(
            activity_df,
            x='Month',
            y='Commits',
            color='Company',
            title='Monthly Development Activity'
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    logger.info("Visualizations complete")

def main():
    st.set_page_config(layout="wide", page_title="GitHub Analytics Dashboard")
    st.title("GitHub Analytics Dashboard")
    
    st.sidebar.header("Settings")
    github_token = st.sidebar.text_input("GitHub Token", type="password")
    uploaded_file = st.sidebar.file_uploader("Company List (.txt)", type="txt")

    if github_token and uploaded_file:
        try:
            github_api = GitHubAPI(github_token)
            companies = parse_company_urls(uploaded_file.read())
            
            if not companies:
                logger.error("No valid company names found")
                st.error("No valid company names found in the file")
                return

            logger.info(f"Starting analysis for {len(companies)} companies")
            
            companies_data = []
            total_progress = st.progress(0)
            company_status = st.empty()

            for idx, company in enumerate(companies, 1):
                company_status.markdown(f"### Analyzing {company} ({idx}/{len(companies)})")
                company_data = get_organization_data(github_api, company)
                
                if company_data:
                    companies_data.append(company_data)
                    logger.info(f"Completed analysis for {company}")
                else:
                    logger.error(f"Failed to analyze {company}")
                
                total_progress.progress(idx / len(companies))

            total_progress.empty()
            company_status.empty()

            if companies_data:
                create_visualizations(companies_data)
                logger.info("Dashboard ready")
            else:
                msg = "No valid data retrieved"
                logger.error(msg)
                st.error(msg)

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please provide your GitHub token and company list to begin.")
        
        with st.expander("Help"):
            st.markdown("""
            ### Instructions
            
            1. **GitHub Token**
               - Go to GitHub.com → Settings → Developer settings
               - Personal access tokens → Tokens (classic)
               - Generate new token with these permissions:
                 - repo
                 - read:org
                 - read:user
               
            2. **Company List**
               - Create a text file (.txt)
               - Add one company per line
               - Use organization names or full GitHub URLs:
               ```
               microsoft
               google
               https://github.com/facebook
               ```
            """)

if __name__ == "__main__":
    main()
