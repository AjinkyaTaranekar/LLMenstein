import json
import os
import re
from typing import Any, Dict, List

import requests


def post_review_comments(all_reviews: Dict[str, Any]) -> requests.Response:
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/pulls/{os.environ['PR_NUMBER']}/reviews"
    headers = {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Create individual review comments for each file
    comments = []
    summary_body = "## 🤖 AI-Powered Code Review Summary\n\n"

    for file_name, review in all_reviews.items():
        file_body = f"### 📄 {file_name} - AI Analysis\n\n"
        if "error" in review:
            file_body += f"❌ **Error reviewing {file_name}:** {review['error']}\n\n"
        else:
            file_body += f"**🔍 Overall Assessment:** {review['general_assessment']}\n\n"

            file_body += "**👍 Positive Aspects:**\n"
            for aspect in review["positive_aspects"]:
                file_body += f"- {aspect}\n"
            file_body += "\n"

            file_body += "**⚠️ Issues:**\n"
            for issue in review["issues"]:
                file_body += f"- **Severity:** {issue['severity']} (Line {issue['line']}): {issue['description']}\n"
                file_body += f"  **Suggestion:** {issue['suggestion']}\n\n"

            file_body += "**📋 Checklist Violations:**\n"
            for violation in review["checklist_violations"]:
                file_body += f"- **Item:** {violation['item']}\n"
                file_body += f"  **Explanation:** {violation['explanation']}\n"
                file_body += f"  **Recommendation:** {violation['recommendation']}\n\n"

        comments.append({"path": file_name, "body": file_body.strip(), "position": 1})

        # Add a brief summary to the main review body
        summary_body += (
            f"- **{file_name}**: {review.get('general_assessment', 'Review failed')}\n"
        )

    summary_body += (
        "\n> Please review the individual file comments for detailed feedback."
    )

    payload = {
        "body": summary_body.strip(),
        "event": "REQUEST_CHANGES",
        "comments": comments,
    }

    print(f"💬 Review comments payload: {json.dumps(payload, indent=2)}")
    response = requests.post(url, headers=headers, json=payload)
    return response


def get_checklist() -> str:
    def clean_markdown(content: str) -> str:
        # Remove URLs
        content = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            content,
        )

        # Remove emphasis markers, but keep the text
        content = re.sub(r"([*_~`])(.+?)\1", r"\2", content)

        # Remove heading markers, but keep the text
        content = re.sub(r"^#+\s+(.+)$", r"\1", content, flags=re.MULTILINE)

        # Replace multiple newlines with double newline, but preserve table structure
        lines = content.split("\n")
        cleaned_lines = []
        in_table = False
        for line in lines:
            if line.strip().startswith("|") and line.strip().endswith("|"):
                in_table = True
                cleaned_lines.append(line)
            elif in_table and line.strip() == "":
                in_table = False
                cleaned_lines.append(line)
            elif not in_table:
                cleaned_lines.append(line.strip())
        content = "\n".join(cleaned_lines)
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Remove any remaining special characters, except those used in tables
        content = re.sub(r"[^\w\s\.\,\;\:\-\(\)\[\]|]", " ", content)

        # Normalize whitespace within lines, but preserve newlines
        content = "\n".join(" ".join(line.split()) for line in content.split("\n"))

        return content.strip()

    def parse_response(response: Dict[str, Any]) -> List[str]:
        parsed_data = []

        def extract_info(item: Dict[str, Any]):
            data = clean_markdown(item.get("content"))

            if data:
                parsed_data.append(data)
            if "pages" in item and isinstance(item["pages"], list):
                for page in item["pages"]:
                    extract_info(page)

        if isinstance(response, dict):
            extract_info(response)
        else:
            raise ValueError("⚠️ The response is not a dict.")

        return parsed_data

    def get_clickup_docs(workspace_id: str, doc_id: str, page_id: str) -> str:
        url = f"https://api.clickup.com/api/v3/workspaces/{workspace_id}/docs/{doc_id}/pages/{page_id}"
        headers = {"Authorization": os.environ.get("CLICKUP_TOKEN")}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            documents = parse_response(data)
        except requests.exceptions.RequestException as req_err:
            print(f"❌ Error during request: {req_err}")
            documents = []
        except ValueError as val_err:
            print(f"❌ Error processing response: {val_err}")
            documents = []

        return "\n\n\n".join(documents)

    checklist_api_url = os.environ.get("CHECKLIST_API_URL", "")
    workspace, doc_ids = checklist_api_url.split("app.clickup.com")[1].split("/v/dc/")
    doc_id, page_id = doc_ids.split("/")
    return get_clickup_docs(workspace, doc_id, page_id)


def call_ollama_api(diff: str, checklist: str, retries: int = 5) -> Dict[str, Any]:
    url = os.environ.get("OLLAMA_API_URL", "") + "/api/generate"
    files = parse_diff(diff)
    all_reviews = {}

    for file_name, file_diff in files.items():
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a highly experienced Senior Code Reviewer. Your task is to review a code diff for a single file and provide constructive feedback to help developers improve their skills. Use the following information and instructions:
            
            REVIEW PROCESS:
            - Analyze the code diff thoroughly for the given file.
            - Check for compliance with the provided checklist.
            - Identify both issues and good practices.
            - Prioritize findings by severity (Critical, Major, Minor).
            - Apply clean code principles and best practices.

            FOR EACH ISSUE:
            - Location: Specify the line number within the file.
            - Explanation: Describe why it's problematic and its potential impact.
            - Suggestion: Offer a specific improvement, including code examples where appropriate.
            - Severity: Categorize as Critical, Major, or Minor.

            ADDITIONAL GUIDELINES:
            - Be thorough but constructive. Aim to educate, not discourage.
            - Consider performance, readability, maintainability, and best practices.
            - If the checklist seems irrelevant to the changes, focus on common issues like null pointer exceptions, naming conventions, code cleanliness, and appropriate design patterns.
            - Highlight any particularly good code practices you notice.
            - Consider the specific context of the file being reviewed.

            Review the following file: {file_name}\n\nChecklist:\n{checklist}\n\nDiff:\n{file_diff}

            OUTPUT FORMAT:
            Based on the provided code diff and checklist for a single file, generate a comprehensive code review following the above format and guidelines.
            Provide your review as a JSON object with the following structure (Provide only a valid JSON, nothing else):
            {{
                "file_name": "path/to/file.ext",
                "general_assessment": "Overall evaluation of the code quality and main areas for improvement",
                "positive_aspects": [
                    "List of good practices or well-written parts of the code"
                ],
                "issues": [
                    {{
                        "severity": "Critical/Major/Minor",
                        "line": line_number,
                        "description": "Detailed explanation of the issue",
                        "suggestion": "Specific improvement recommendation, including code example if applicable"
                    }}
                ],
                "checklist_violations": [
                    {{
                        "item": "Specific checklist item that was violated",
                        "explanation": "Why this item was not followed and its importance",
                        "recommendation": "How to address this violation"
                    }}
                ]
            }}
            EXAMPLE OUTPUT (truncated for brevity):
            {{
                "file_name": "main.py",
                "general_assessment": "The code shows a good understanding of basic concepts, but there are several areas for improvement in terms of error handling and code organization.",
                "positive_aspects": [
                    "Consistent naming convention for variables",
                    "Good use of comments to explain complex logic"
                ],
                "issues": [
                    {{
                        "severity": "Major",
                        "line": 23,
                        "description": "Potential null pointer exception. The 'user' object is not checked for null before accessing its properties.",
                        "suggestion": "Add a null check before accessing 'user' properties. Example: if user is not None:"
                    }}
                ],
                "checklist_violations": [
                    {{
                        "item": "Error handling",
                        "explanation": "The code lacks proper error handling in several critical sections.",
                        "recommendation": "Implement try-except blocks for potential exceptions, especially in file operations and network calls."
                    }}
                ]
            }}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """

        payload = {"model": "llama3.1", "prompt": prompt, "stream": False}

        print(f"Processing file review: {file_name}")
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                llm_response = response.json().get("response", "")
                print("Review received: ", llm_response)
                review = json.loads(llm_response)
                all_reviews[file_name] = review
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"Attempt {attempt + 1} failed for {file_name}: {str(e)}")
                if attempt == retries - 1:
                    print(f"❌ All retries exhausted for {file_name}")
                    all_reviews[file_name] = {
                        "error": f"Failed to get a valid response after {retries} attempts"
                    }

    return all_reviews


def parse_diff(diff_text: str) -> Dict[str, str]:
    files = {}
    current_file = None
    current_content = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file:
                files[current_file] = "\n".join(current_content)
            current_file = line.split()[-1].lstrip("b/")
            current_content = []
        else:
            current_content.append(line)

    if current_file:
        files[current_file] = "\n".join(current_content)

    return files


def get_pr_diff(pr_url: str, headers: Dict[str, str]) -> str:
    response = requests.get(pr_url, headers=headers)
    response.raise_for_status()
    return response.text


def main():
    pr_url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/pulls/{os.environ['PR_NUMBER']}"
    headers = {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3.diff",
    }

    try:
        diff = get_pr_diff(pr_url, headers)
        print("🔍 Diff received:\n" + "-" * 100)
        print(diff)

        checklist = get_checklist()
        print("📋 Checklist fetched:\n" + "-" * 100)
        print(checklist)

        review = call_ollama_api(diff, checklist)
        print("✅ Review received:\n" + "-" * 100)
        print(json.dumps(review, indent=2))

        res = post_review_comments(review)
        print("📤 GitHub response:\n" + "-" * 100)
        print(res.json())

    except requests.RequestException as e:
        print(f"❌ Error occurred while making a request: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error occurred while parsing JSON: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
