import tableauserverclient as TSC
import json

# --- Load Configuration from config.json ---
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

TABLEAU_SERVER_URL = config['tableau_server_url']
TABLEAU_SITE_NAME = config['tableau_site_name']
PAT_NAME = config['pat_name']
PAT_SECRET = config['pat_secret']
PROJECT_NAME = config['project_name']


def main():
    """
    Connects to a Tableau project, and extracts and prints permissions
    for each workbook.
    """
    tableau_auth = TSC.PersonalAccessTokenAuth(PAT_NAME, PAT_SECRET, site_id=TABLEAU_SITE_NAME)
    server = TSC.Server(TABLEAU_SERVER_URL, use_server_version=True)

    with server.auth.sign_in(tableau_auth):
        print(f"Successfully signed in to {server.server_info.product_version}")

        # Find the project
        all_projects, _ = server.projects.get()
        project = next((p for p in all_projects if p.name == PROJECT_NAME), None)

        if not project:
            print(f"Project '{PROJECT_NAME}' not found.")
            return

        print(f"Found project: {project.name}")

        # To get workbook permissions, we need to populate the workbooks for the project first
        server.projects.populate_workbooks(project)

        # Get permissions for each workbook
        for workbook in project.workbooks:
            server.workbooks.populate_permissions(workbook)
            print(f"\n--- Workbook: {workbook.name} ---")

            for rule in workbook.permissions:
                grantee = rule.grantee
                capability_name = rule.capabilities
                print(f"  - Grantee: {grantee.tag_name} '{grantee.id}' has permissions: {capability_name}")


if __name__ == "__main__":
    main()
