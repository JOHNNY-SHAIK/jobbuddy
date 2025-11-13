import pandas as pd
import os

def create_jobs_csv(path="data/jobs_sample.csv"):
    jobs = [
        {"job_id": 1, "title": "Junior Data Scientist", "skills": "python, pandas, numpy, machine learning, SQL", "description": "Work with datasets, build ML models, data cleaning and visualization."},
        {"job_id": 2, "title": "Backend Developer (Python)", "skills": "python, django, REST, SQL, docker", "description": "Develop REST APIs, work with databases and deploy using Docker."},
        {"job_id": 3, "title": "Frontend Developer", "skills": "javascript, react, html, css", "description": "Build responsive web UIs using React and modern JS."},
        {"job_id": 4, "title": "Data Analyst Intern", "skills": "excel, sql, tableau, python, pandas", "description": "Perform reporting, dashboards, and data analysis."},
        {"job_id": 5, "title": "Machine Learning Engineer", "skills": "python, tensorflow, pytorch, machine learning, devops", "description": "Productionize ML models and collaborate with dev teams."},
        {"job_id": 6, "title": "QA Engineer", "skills": "manual testing, selenium, python, test plans", "description": "Write test cases and automate UI tests with Selenium."},
        {"job_id": 7, "title": "Cloud Engineer", "skills": "aws, terraform, docker, kubernetes", "description": "Manage infra, write IaC and support deployments."},
        {"job_id": 8, "title": "Business Analyst", "skills": "excel, sql, powerpoint, stakeholder management", "description": "Gather requirements, create reports, and help stakeholders."}
    ]
    df = pd.DataFrame(jobs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Jobs saved to {path}")

def create_users_csv(path="data/users_sample.csv"):
    users = [
        {"user_id": 1, "name": "Alice", "title": "Student - Data Science", "skills": "python, pandas, numpy, SQL", "bio": "Final year student, built small ML projects and internships."},
        {"user_id": 2, "name": "Bob", "title": "Backend Developer", "skills": "python, django, docker", "bio": "2 years experience building APIs."},
        {"user_id": 3, "name": "Charlie", "title": "Frontend Learner", "skills": "javascript, react, html", "bio": "Learning React, looking for frontend internships."}
    ]
    df = pd.DataFrame(users)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Users saved to {path}")

if __name__ == "__main__":
    create_jobs_csv()
    create_users_csv()
