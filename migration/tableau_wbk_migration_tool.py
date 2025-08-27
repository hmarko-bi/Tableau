import requests
import xml.etree.ElementTree as ET
import os
import zipfile
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import json
import uuid
import mimetypes
import tempfile

# Configure logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workbook_migration.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

API_VERSION = "3.25"

@dataclass
class TableauSite:
    """Configuration for a Tableau site"""
    server: str
    site: str
    pat_name: str
    pat_secret: str
    project: str  # Used for datasources; workbooks use different projects

@dataclass
class MigrationResult:
    """Result of a single workbook migration"""
    name: str
    success: bool
    error: Optional[str] = None
    file_path: Optional[str] = None

class TableauAPIClient:
    """Tableau API client for workbook migration"""
    
    def __init__(self, server: str, api_version: str = API_VERSION):
        self.server = server.rstrip('/')
        self.api_version = api_version
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/xml',
            'User-Agent': 'TableauWorkbookMigrationTool/1.0'
        })
        self.token = None
        self.site_id = None
        self.user_id = None
        
    def sign_in(self, site: str, pat_name: str, pat_secret: str) -> Tuple[str, str, str]:
        """Sign in using Personal Access Token"""
        url = f"{self.server}/api/{self.api_version}/auth/signin"
        xml_payload = f"""<tsRequest>
            <credentials personalAccessTokenName="{pat_name}" personalAccessTokenSecret="{pat_secret}">
                <site contentUrl="{site}" />
            </credentials>
        </tsRequest>"""
        
        try:
            logger.debug(f"Attempting to sign in to {site} on {self.server}")
            response = self.session.post(
                url, 
                data=xml_payload,
                headers={'Content-Type': 'application/xml'},
                timeout=60
            )
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            credentials = root.find('.//credentials')
            if credentials is None:
                ns = {'t': 'http://tableau.com/api'}
                credentials = root.find('.//t:credentials', ns)
            if credentials is None and root.tag.endswith('credentials'):
                credentials = root
            if credentials is None:
                raise ValueError(f"Invalid response: credentials element not found. Response: {response.text}")
                
            self.token = credentials.get('token')
            site_elem = credentials.find('.//site') or credentials.find('.//t:site', {'t': 'http://tableau.com/api'})
            user_elem = credentials.find('.//user') or credentials.find('.//t:user', {'t': 'http://tableau.com/api'})
            
            if not self.token or site_elem is None or user_elem is None:
                raise ValueError(f"Authentication failed: missing required elements. Response: {response.text}")
                
            self.site_id = site_elem.get('id')
            self.user_id = user_elem.get('id')
            
            if not all([self.token, self.site_id, self.user_id]):
                raise ValueError(f"Authentication failed: missing required values")
            
            self.session.headers.update({'X-Tableau-Auth': self.token})
            logger.info(f"Successfully signed in to {site} on {self.server}")
            return self.token, self.site_id, self.user_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during sign-in: {e}")
            raise
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Sign-in failed: {e}")
            raise

    def sign_out(self) -> None:
        """Sign out with session cleanup"""
        if self.token:
            try:
                url = f"{self.server}/api/{self.api_version}/auth/signout"
                self.session.post(url, timeout=10)
                logger.info("Successfully signed out")
            except Exception as e:
                logger.warning(f"Error during sign-out: {e}")
            finally:
                self.token = None
                self.site_id = None
                self.user_id = None
                if 'X-Tableau-Auth' in self.session.headers:
                    del self.session.headers['X-Tableau-Auth']

    def get_project_id(self, project_name: str) -> str:
        """Get project ID by name"""
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/projects"
        try:
            logger.debug(f"Fetching project ID for '{project_name}'")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {'t': 'http://tableau.com/api'}
            for project in root.findall('.//t:project', ns):
                if project.get('name') == project_name:
                    project_id = project.get('id')
                    logger.info(f"Found project '{project_name}' with ID: {project_id}")
                    return project_id
            raise ValueError(f"Project '{project_name}' not found")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching projects: {e}")
            raise

    def get_workbooks_in_project(self, project_id: str) -> List[Dict[str, str]]:
        """Get all workbooks in a project with pagination"""
        workbooks = []
        page_number = 1
        page_size = 100
        while True:
            url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/workbooks"
            params = {'pageSize': page_size, 'pageNumber': page_number}
            try:
                logger.debug(f"Fetching workbooks page {page_number}")
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                ns = {'t': 'http://tableau.com/api'}
                page_workbooks = []
                for wb in root.findall('.//t:workbook', ns):
                    project_elem = wb.find('t:project', ns)
                    if project_elem is not None and project_elem.get('id') == project_id:
                        wb_info = {
                            'id': wb.get('id'),
                            'name': wb.get('name'),
                            'content_url': wb.get('contentUrl', ''),
                            'created_at': wb.get('createdAt', ''),
                            'updated_at': wb.get('updatedAt', '')
                        }
                        page_workbooks.append(wb_info)
                workbooks.extend(page_workbooks)
                pagination = root.find('.//t:pagination', ns)
                if pagination is not None:
                    total_available = int(pagination.get('totalAvailable', 0))
                    if page_number * page_size >= total_available:
                        break
                    page_number += 1
                else:
                    break
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching workbooks (page {page_number}): {e}")
                raise
        logger.info(f"Found {len(workbooks)} workbooks in project")
        return workbooks

    def get_datasource_id(self, project_id: str, datasource_name: str) -> str:
        """Get datasource ID by name in a specific project"""
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/datasources"
        try:
            logger.debug(f"Fetching datasource ID for '{datasource_name}' in project {project_id}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {'t': 'http://tableau.com/api'}
            for ds in root.findall('.//t:datasource', ns):
                project_elem = ds.find('t:project', ns)
                if project_elem is not None and project_elem.get('id') == project_id and ds.get('name') == datasource_name:
                    ds_id = ds.get('id')
                    logger.info(f"Found datasource '{datasource_name}' with ID: {ds_id}")
                    return ds_id
            raise ValueError(f"Datasource '{datasource_name}' not found in project {project_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching datasources: {e}")
            raise

    def download_workbook(self, workbook_id: str, output_path: str) -> str:
        """Download workbook with progress tracking"""
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/workbooks/{workbook_id}/content"
        try:
            logger.debug(f"Starting download of workbook {workbook_id} to {output_path}")
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            content_disposition = response.headers.get('Content-Disposition', '')
            if '.twbx' in content_disposition:
                file_path = f"{output_path}.twbx"
            elif '.twb' in content_disposition:
                file_path = f"{output_path}.twb"
            else:
                file_path = f"{output_path}.twbx"
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"Downloading {workbook_id}: {downloaded}/{total_size} bytes ({percent:.1f}%)")
            logger.info(f"Downloaded {downloaded} bytes to {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading workbook {workbook_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading workbook {workbook_id}: {e}")
            raise

    def publish_workbook(self, project_id: str, file_path: str, workbook_name: str, 
                        overwrite: bool = True) -> str:
        """Publish workbook using multipart/mixed content type"""
        wb_type = os.path.splitext(file_path)[1].lstrip('.')
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/workbooks"
        params = {'workbookType': wb_type}
        if overwrite:
            params['overwrite'] = 'true'
        xml_payload = f"""<tsRequest>
            <workbook name="{workbook_name}" showTabs="true">
                <project id="{project_id}" />
            </workbook>
        </tsRequest>"""
        boundary = f"----TableauMultipartBoundary{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        body = []
        body.append(f"--{boundary}")
        body.append('Content-Type: text/xml; charset=UTF-8')
        body.append('Content-Disposition: name="request_payload"')
        body.append('')
        body.append(xml_payload)
        body.append(f"--{boundary}")
        body.append(f'Content-Type: {mime_type}')
        body.append(f'Content-Disposition: name="tableau_workbook"; filename="{os.path.basename(file_path)}"')
        body.append('')
        with open(file_path, 'rb') as f:
            body.append(f.read().decode('latin1'))
        body.append(f"--{boundary}--")
        body = '\r\n'.join(body) if isinstance(body[-1], str) else '\r\n'.join(body[:-1]) + '\r\n' + body[-1] + '\r\n'
        headers = {
            'Content-Type': f'multipart/mixed; boundary={boundary}',
            'X-Tableau-Auth': self.token
        }
        try:
            logger.debug(f"Publishing workbook {workbook_name} to project {project_id}")
            original_accept = self.session.headers.pop('Accept', None)
            response = self.session.post(url, params=params, data=body.encode('latin1'), headers=headers, timeout=600)
            if original_accept:
                self.session.headers['Accept'] = original_accept
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {'t': 'http://tableau.com/api'}
            wb_elem = root.find('.//t:workbook', ns)
            if wb_elem is not None:
                published_id = wb_elem.get('id')
                logger.info(f"Successfully published '{workbook_name}' with ID: {published_id}")
                return published_id
            else:
                logger.error(f"Could not find workbook details in publish response: {response.text}")
                raise ValueError("Published workbook ID not found in response")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error publishing workbook '{workbook_name}': {e}. Response: {e.response.text if e.response else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error publishing workbook '{workbook_name}': {e}")
            raise

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sign_out()

class TableauMigrationTool:
    """Tool for migrating Tableau workbooks with datasource switching"""
    
    def __init__(self, source_config: TableauSite, target_config: TableauSite, 
                 source_project_name: str, target_project_name: str, 
                 max_workers: int = 2, temp_dir: str = "temp_downloads"):
        self.source_config = source_config
        self.target_config = target_config
        self.source_project_name = source_project_name
        self.target_project_name = target_project_name
        self.max_workers = max_workers
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.last_request_time = 0
        self.min_request_interval = 0.2
        
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            logger.debug(f"Rate limiting: waiting {self.min_request_interval - elapsed:.2f} seconds")
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def update_workbook_datasource(self, workbook_path: str, source_project_id: str, target_project_id: str, target_client: TableauAPIClient) -> str:
        """Update workbook XML to point to target site's datasources"""
        try:
            logger.debug(f"Updating datasource references in {workbook_path}")
            file_ext = os.path.splitext(workbook_path)[1].lower()
            with tempfile.TemporaryDirectory() as temp_dir:
                if file_ext == '.twbx':
                    with zipfile.ZipFile(workbook_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    twb_files = list(Path(temp_dir).glob('*.twb'))
                    if not twb_files:
                        raise ValueError("No .twb file found in .twbx archive")
                    twb_path = twb_files[0]
                else:
                    twb_path = Path(workbook_path)
                
                tree = ET.parse(twb_path)
                root = tree.getroot()
                ns = {'t': 'http://tableau.com/xml'}
                datasources = root.findall('.//t:datasource', ns)
                modified = False
                for ds in datasources:
                    ds_name = ds.get('name')
                    if ds_name:
                        try:
                            new_ds_id = target_client.get_datasource_id(target_project_id, ds_name)
                            connection = ds.find('t:connection', ns)
                            if connection is not None:
                                connection.set('server', target_client.server)
                                connection.set('site', target_client.site_id)
                                ds.set('id', new_ds_id)
                                modified = True
                                logger.debug(f"Updated datasource '{ds_name}' to target ID {new_ds_id}")
                        except ValueError as e:
                            logger.warning(f"Could not update datasource '{ds_name}': {e}")
                
                if modified:
                    if file_ext == '.twbx':
                        new_twb_path = Path(temp_dir) / twb_path.name
                        tree.write(new_twb_path)
                        new_workbook_path = f"{os.path.splitext(workbook_path)[0]}_updated.twbx"
                        with zipfile.ZipFile(new_workbook_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    if file.endswith('.twb'):
                                        zip_ref.write(os.path.join(root, file), file)
                                    else:
                                        zip_ref.write(os.path.join(root, file), os.path.join(root[len(temp_dir)+1:], file))
                    else:
                        new_workbook_path = f"{os.path.splitext(workbook_path)[0]}_updated.twb"
                        tree.write(new_workbook_path)
                    logger.info(f"Updated workbook saved to {new_workbook_path}")
                    return new_workbook_path
                else:
                    logger.info(f"No datasource updates needed for {workbook_path}")
                    return workbook_path
        except Exception as e:
            logger.error(f"Failed to update workbook {workbook_path}: {e}")
            raise

    def download_workbook(self, source_client: TableauAPIClient, wb_info: Dict[str, str]) -> MigrationResult:
        """Download a single workbook"""
        try:
            self._rate_limit()
            temp_file = self.temp_dir / f"temp_{wb_info['name']}"
            logger.debug(f"Downloading workbook {wb_info['name']} (ID: {wb_info['id']})")
            file_path = source_client.download_workbook(wb_info['id'], str(temp_file))
            return MigrationResult(
                name=wb_info['name'],
                success=True,
                file_path=file_path
            )
        except Exception as e:
            logger.error(f"Failed to download {wb_info['name']}: {e}")
            return MigrationResult(
                name=wb_info['name'],
                success=False,
                error=str(e)
            )

    def publish_workbook(self, target_client: TableauAPIClient, target_project_id: str, 
                        result: MigrationResult, source_project_id: str, target_ds_project_id: str) -> MigrationResult:
        """Upload a single workbook after updating datasources"""
        if not result.success or not result.file_path:
            logger.debug(f"Skipping publish for {result.name} due to previous failure")
            return result
        try:
            self._rate_limit()
            updated_file_path = self.update_workbook_datasource(result.file_path, source_project_id, target_ds_project_id, target_client)
            logger.debug(f"Publishing {result.name} to target project {target_project_id}")
            target_client.publish_workbook(target_project_id, updated_file_path, result.name)
            return MigrationResult(
                name=result.name,
                success=True,
                file_path=updated_file_path
            )
        except Exception as e:
            logger.error(f"Failed to publish {result.name}: {e}")
            return MigrationResult(
                name=result.name,
                success=False,
                error=str(e),
                file_path=result.file_path
            )

    def migrate_workbooks(self, workbook_names: List[str] = None) -> List[MigrationResult]:
        """Migrate workbooks with datasource switching"""
        results = []
        downloaded_files = []
        try:
            logger.info("Connecting to source site")
            with TableauAPIClient(self.source_config.server) as source_client:
                source_client.sign_in(
                    self.source_config.site,
                    self.source_config.pat_name,
                    self.source_config.pat_secret
                )
                logger.info(f"Fetching project ID for '{self.source_project_name}'")
                source_project_id = source_client.get_project_id(self.source_project_name)
                logger.info("Fetching available workbooks")
                available_workbooks = source_client.get_workbooks_in_project(source_project_id)
                if workbook_names:
                    available_workbooks = [
                        wb for wb in available_workbooks 
                        if wb['name'] in workbook_names
                    ]
                if not available_workbooks:
                    logger.warning("No workbooks found to migrate")
                    return results
                logger.info(f"Starting download of {len(available_workbooks)} workbooks")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    download_futures = {
                        executor.submit(self.download_workbook, source_client, wb_info): wb_info
                        for wb_info in available_workbooks
                    }
                    for future in as_completed(download_futures, timeout=3600):
                        try:
                            result = future.result()
                            results.append(result)
                            if result.success:
                                downloaded_files.append(result)
                            logger.info(f"Download completed for {result.name}: {'Success' if result.success else 'Failed'}")
                        except TimeoutError:
                            wb_info = download_futures[future]
                            logger.error(f"Download timeout for {wb_info['name']} after 3600 seconds")
                            results.append(MigrationResult(
                                name=wb_info['name'],
                                success=False,
                                error="Download timeout after 3600 seconds"
                            ))
            logger.info("Connecting to target site")
            with TableauAPIClient(self.target_config.server) as target_client:
                target_client.sign_in(
                    self.target_config.site,
                    self.target_config.pat_name,
                    self.target_config.pat_secret
                )
                logger.info(f"Fetching project ID for '{self.target_project_name}'")
                target_project_id = target_client.get_project_id(self.target_project_name)
                logger.info(f"Fetching datasource project ID for '{self.target_config.project}'")
                target_ds_project_id = target_client.get_project_id(self.target_config.project)
                logger.info(f"Starting upload of {len(downloaded_files)} workbooks")
                for download_result in downloaded_files:
                    upload_result = self.publish_workbook(target_client, target_project_id, download_result, source_project_id, target_ds_project_id)
                    for i, result in enumerate(results):
                        if result.name == upload_result.name:
                            results[i] = upload_result
                            break
                    logger.info(f"Upload completed for {upload_result.name}: {'Success' if upload_result.success else 'Failed'}")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            self._cleanup_temp_files(results)
        return results

    def _cleanup_temp_files(self, results: List[MigrationResult]):
        """Clean up temporary files"""
        for result in results:
            if result.file_path and os.path.exists(result.file_path):
                try:
                    os.remove(result.file_path)
                    logger.debug(f"Cleaned up temporary file: {result.file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {result.file_path}: {e}")

    def generate_report(self, results: List[MigrationResult]) -> Dict:
        """Generate migration report"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        report = {
            'total_workbooks': len(results),
            'successful_migrations': len(successful),
            'failed_migrations': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'successful_workbooks': [r.name for r in successful],
            'failed_workbooks': [{'name': r.name, 'error': r.error} for r in failed]
        }
        return report

def main():
    """Main execution function"""
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Configuration file 'config.json' not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'config.json': {e}")
        raise
    source_config = TableauSite(**config['source_config'])
    target_config = TableauSite(**config['target_config'])
    try:
        with open('workbooks.json', 'r') as f:
            workbooks_to_migrate = json.load(f)
    except FileNotFoundError:
        logger.error("Workbooks file 'workbooks.json' not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'workbooks.json': {e}")
        raise
    migration_tool = TableauMigrationTool(
        source_config=source_config,
        target_config=target_config,
        source_project_name="Production",
        target_project_name="AZ Workbook",
        max_workers=2
    )
    try:
        logger.info("Starting Tableau workbook migration")
        results = migration_tool.migrate_workbooks(workbooks_to_migrate)
        report = migration_tool.generate_report(results)
        logger.info("Migration completed!")
        logger.info(f"Total workbooks: {report['total_workbooks']}")
        logger.info(f"Successful: {report['successful_migrations']}")
        logger.info(f"Failed: {report['failed_migrations']}")
        logger.info(f"Success rate: {report['success_rate']:.1f}%")
        if report['failed_workbooks']:
            logger.info("Failed workbooks:")
            for failed in report['failed_workbooks']:
                logger.info(f"  - {failed['name']}: {failed['error']}")
        with open('workbook_migration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()