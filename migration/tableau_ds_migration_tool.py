import requests
import xml.etree.ElementTree as ET
import os
import zipfile
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import uuid
import mimetypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tableau_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use a more conservative API version that's known to work
API_VERSION = "3.25"  # Using the version from your original script

@dataclass
class TableauSite:
    """Configuration for a Tableau site"""
    server: str
    site: str
    pat_name: str
    pat_secret: str
    project: str
    
@dataclass
class MigrationResult:
    """Result of a single datasource migration"""
    name: str
    success: bool
    error: Optional[str] = None
    file_path: Optional[str] = None

class TableauAPIClient:
    """Enhanced Tableau API client with modern practices"""
    
    def __init__(self, server: str, api_version: str = API_VERSION):
        self.server = server.rstrip('/')
        self.api_version = api_version
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/xml',
            'User-Agent': 'TableauMigrationTool/2.0'
        })
        self.token = None
        self.site_id = None
        self.user_id = None
        
    def sign_in(self, site: str, pat_name: str, pat_secret: str) -> Tuple[str, str, str]:
        """Sign in using Personal Access Token with improved error handling"""
        url = f"{self.server}/api/{self.api_version}/auth/signin"
        
        # Use f-string for XML template (more readable)
        xml_payload = f"""<tsRequest>
            <credentials personalAccessTokenName="{pat_name}" personalAccessTokenSecret="{pat_secret}">
                <site contentUrl="{site}" />
            </credentials>
        </tsRequest>"""
        
        try:
            response = self.session.post(
                url, 
                data=xml_payload,
                headers={'Content-Type': 'application/xml'},
                timeout=30
            )
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Debug: Log the response to understand structure
            logger.debug(f"Authentication response: {response.text}")
            
            if not response.text.strip():
                raise ValueError("Empty response from authentication endpoint")
            
            # Parse XML response
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                raise ValueError(f"Invalid XML response: {e}. Response: {response.text}")
            
            # Try multiple approaches to find credentials
            credentials = None
            
            # Method 1: Direct search without namespace
            credentials = root.find('.//credentials')
            
            # Method 2: Search with namespace
            if credentials is None:
                ns = {'t': 'http://tableau.com/api'}
                credentials = root.find('.//t:credentials', ns)
            
            # Method 3: Check if root is credentials
            if credentials is None and root.tag.endswith('credentials'):
                credentials = root
                
            if credentials is None:
                raise ValueError(f"Invalid response: credentials element not found. Response: {response.text}")
                
            self.token = credentials.get('token')
            
            # Find site and user elements with fallback approaches
            site_elem = None
            user_elem = None
            
            # Try different ways to find site element
            site_elem = credentials.find('.//site')
            if site_elem is None:
                ns = {'t': 'http://tableau.com/api'}
                site_elem = credentials.find('.//t:site', ns)
            
            # Try different ways to find user element  
            user_elem = credentials.find('.//user')
            if user_elem is None:
                ns = {'t': 'http://tableau.com/api'}
                user_elem = credentials.find('.//t:user', ns)
            
            # Fix deprecation warning by checking elements properly
            if not self.token or site_elem is None or user_elem is None:
                raise ValueError(f"Authentication failed: missing required elements. Token: {bool(self.token)}, Site: {site_elem is not None}, User: {user_elem is not None}. Full response: {response.text}")
                
            self.site_id = site_elem.get('id')
            self.user_id = user_elem.get('id')
            
            # Validate we got all required values
            if not all([self.token, self.site_id, self.user_id]):
                raise ValueError(f"Authentication failed: missing required values. Token: {bool(self.token)}, Site ID: {self.site_id}, User ID: {self.user_id}")
            
            # Update session headers
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
        """Sign out with proper session cleanup"""
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
        """Get project ID with improved filtering"""
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/projects"
        
        try:
            response = self.session.get(url, timeout=30)
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

    def get_datasources_in_project(self, project_id: str) -> List[Dict[str, str]]:
        """Get all datasources in a project with pagination"""
        datasources = []
        page_number = 1
        page_size = 100
        
        while True:
            url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/datasources"
            params = {
                'pageSize': page_size,
                'pageNumber': page_number,
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                ns = {'t': 'http://tableau.com/api'}
                
                page_datasources = []
                for ds in root.findall('.//t:datasource', ns):
                    project_elem = ds.find('t:project', ns)
                    if project_elem is not None and project_elem.get('id') == project_id:
                        ds_info = {
                            'id': ds.get('id'),
                            'name': ds.get('name'),
                            'content_url': ds.get('contentUrl', ''),
                            'created_at': ds.get('createdAt', ''),
                            'updated_at': ds.get('updatedAt', '')
                        }
                        page_datasources.append(ds_info)
                
                datasources.extend(page_datasources)
                
                # Check pagination
                pagination = root.find('.//t:pagination', ns)
                if pagination is not None:
                    total_available = int(pagination.get('totalAvailable', 0))
                    if page_number * page_size >= total_available:
                        break
                    page_number += 1
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching datasources (page {page_number}): {e}")
                raise
                
        logger.info(f"Found {len(datasources)} datasources in project")
        return datasources

    def download_datasource(self, datasource_id: str, output_path: str) -> str:
        """Download datasource with improved error handling and progress tracking"""
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/datasources/{datasource_id}/content"
        
        try:
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Determine file extension from headers
            content_disposition = response.headers.get('Content-Disposition', '')
            if '.tdsx' in content_disposition:
                file_path = f"{output_path}.tdsx"
            elif '.tds' in content_disposition:
                file_path = f"{output_path}.tds"
            else:
                file_path = f"{output_path}.tdsx"  # Default
            
            # Download with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
            logger.info(f"Downloaded {downloaded} bytes to {file_path}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading datasource {datasource_id}: {e}")
            raise

    def publish_datasource(self, project_id: str, file_path: str, datasource_name: str, 
                       overwrite: bool = True) -> str:
        """Publish datasource using multipart/mixed content type for Tableau API."""
        ds_type = os.path.splitext(file_path)[1].lstrip('.')
        url = f"{self.server}/api/{self.api_version}/sites/{self.site_id}/datasources"
        params = {'datasourceType': ds_type}
        if overwrite:
            params['overwrite'] = 'true'

        # XML payload for the request
        xml_payload = f"""<tsRequest>
            <datasource name="{datasource_name}">
                <project id="{project_id}" />
            </datasource>
        </tsRequest>"""

        # Generate a unique boundary
        boundary = f"----TableauMultipartBoundary{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

        # Construct the multipart/mixed payload manually
        body = []
        body.append(f"--{boundary}")
        body.append('Content-Type: text/xml; charset=UTF-8')
        body.append('Content-Disposition: name="request_payload"')
        body.append('')
        body.append(xml_payload)
        body.append(f"--{boundary}")
        body.append(f'Content-Type: {mime_type}')
        body.append(f'Content-Disposition: name="tableau_datasource"; filename="{os.path.basename(file_path)}"')
        body.append('')
        
        # Read the file content
        with open(file_path, 'rb') as f:
            body.append(f.read().decode('latin1'))  # Use latin1 to avoid encoding issues with binary data
        body.append(f"--{boundary}--")
        
        # Join the body parts with newlines
        body = '\r\n'.join(body) if isinstance(body[-1], str) else '\r\n'.join(body[:-1]) + '\r\n' + body[-1] + '\r\n'

        # Set headers for multipart/mixed
        headers = {
            'Content-Type': f'multipart/mixed; boundary={boundary}',
            'X-Tableau-Auth': self.token
        }

        try:
            # Temporarily remove the 'Accept' header for this request
            original_accept = self.session.headers.pop('Accept', None)

            # Send the request with the custom body
            response = self.session.post(url, params=params, data=body.encode('latin1'), headers=headers, timeout=300)

            # Restore the 'Accept' header
            if original_accept:
                self.session.headers['Accept'] = original_accept

            response.raise_for_status()

            # Parse response to get published datasource ID
            root = ET.fromstring(response.content)
            ns = {'t': 'http://tableau.com/api'}
            ds_elem = root.find('.//t:datasource', ns)

            if ds_elem is not None:
                published_id = ds_elem.get('id')
                logger.info(f"Successfully published '{datasource_name}' with ID: {published_id}")
                return published_id
            else:
                logger.error(f"Could not find datasource details in publish response: {response.text}")
                raise ValueError("Published datasource ID not found in response")

        except requests.exceptions.RequestException as e:
            if e.response is not None:
                logger.error(f"Error publishing datasource '{datasource_name}': {e}. Response: {e.response.text}")
            else:
                logger.error(f"Error publishing datasource '{datasource_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during publish of '{datasource_name}': {e}")
            raise
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sign_out()

class TableauMigrationTool:
    """Enhanced migration tool with concurrent processing and better error handling"""
    
    def __init__(self, source_config: TableauSite, target_config: TableauSite, 
                 max_workers: int = 3, temp_dir: str = "temp_downloads"):
        self.source_config = source_config
        self.target_config = target_config
        self.max_workers = max_workers
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Rate limiting to respect API limits
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def download_datasource(self, source_client: TableauAPIClient, ds_info: Dict[str, str]) -> MigrationResult:
        """Download a single datasource"""
        try:
            self._rate_limit()
            
            temp_file = self.temp_dir / f"temp_{ds_info['name']}"
            file_path = source_client.download_datasource(ds_info['id'], str(temp_file))
            
            return MigrationResult(
                name=ds_info['name'],
                success=True,
                file_path=file_path
            )
            
        except Exception as e:
            logger.error(f"Failed to download {ds_info['name']}: {e}")
            return MigrationResult(
                name=ds_info['name'],
                success=False,
                error=str(e)
            )

    def publish_datasource(self, target_client: TableauAPIClient, target_project_id: str, 
                          result: MigrationResult) -> MigrationResult:
        """Upload a single datasource"""
        if not result.success or not result.file_path:
            return result
            
        try:
            self._rate_limit()
            
            target_client.publish_datasource(
                target_project_id, 
                result.file_path, 
                result.name
            )
            
            return MigrationResult(
                name=result.name,
                success=True,
                file_path=result.file_path
            )
            
        except Exception as e:
            logger.error(f"Failed to publish {result.name}: {e}")
            return MigrationResult(
                name=result.name,
                success=False,
                error=str(e),
                file_path=result.file_path
            )

    def migrate_datasources(self, datasource_names: List[str] = None) -> List[MigrationResult]:
        """Migrate datasources with improved error handling and logging"""
        results = []
        downloaded_files = []
        
        try:
            # Connect to source
            with TableauAPIClient(self.source_config.server) as source_client:
                source_client.sign_in(
                    self.source_config.site,
                    self.source_config.pat_name,
                    self.source_config.pat_secret
                )
                
                source_project_id = source_client.get_project_id(self.source_config.project)
                
                # Get available datasources
                available_datasources = source_client.get_datasources_in_project(source_project_id)
                
                # Filter datasources if specific names provided
                if datasource_names:
                    available_datasources = [
                        ds for ds in available_datasources 
                        if ds['name'] in datasource_names
                    ]
                
                if not available_datasources:
                    logger.warning("No datasources found to migrate")
                    return results
                
                logger.info(f"Starting download of {len(available_datasources)} datasources")
                
                # Download datasources (can be done concurrently)
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    download_futures = {
                        executor.submit(self.download_datasource, source_client, ds_info): ds_info
                        for ds_info in available_datasources
                    }
                    
                    for future in as_completed(download_futures):
                        result = future.result()
                        results.append(result)
                        if result.success:
                            downloaded_files.append(result)
                        logger.info(f"Download completed for {result.name}: {'Success' if result.success else 'Failed'}")
                
            # Connect to target and publish
            with TableauAPIClient(self.target_config.server) as target_client:
                target_client.sign_in(
                    self.target_config.site,
                    self.target_config.pat_name,
                    self.target_config.pat_secret
                )
                
                target_project_id = target_client.get_project_id(self.target_config.project)
                
                logger.info(f"Starting upload of {len(downloaded_files)} datasources")
                
                # Upload datasources (sequential for better error handling)
                for download_result in downloaded_files:
                    upload_result = self.publish_datasource(target_client, target_project_id, download_result)
                    
                    # Update the result in the results list
                    for i, result in enumerate(results):
                        if result.name == upload_result.name:
                            results[i] = upload_result
                            break
                    
                    logger.info(f"Upload completed for {upload_result.name}: {'Success' if upload_result.success else 'Failed'}")
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            # Cleanup temporary files
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
            'total_datasources': len(results),
            'successful_migrations': len(successful),
            'failed_migrations': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'successful_datasources': [r.name for r in successful],
            'failed_datasources': [{'name': r.name, 'error': r.error} for r in failed]
        }
        
        return report

def main():
    """Main execution function with configuration"""
    
    # Enable debug logging for troubleshooting
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration from config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Configuration file 'config.json' not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'config.json': {e}")
        raise

    # Create TableauSite objects from config
    source_config = TableauSite(**config['source_config'])
    target_config = TableauSite(**config['target_config'])
    
    # Load datasources from datasources.json
    try:
        with open('datasources.json', 'r') as f:
            datasources_to_migrate = json.load(f)
    except FileNotFoundError:
        logger.error("Datasources file 'datasources.json' not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'datasources.json': {e}")
        raise
    
    # Create migration tool
    migration_tool = TableauMigrationTool(
        source_config=source_config,
        target_config=target_config,
        max_workers=2  # Conservative for API rate limiting
    )
    
    try:
        logger.info("Starting Tableau datasource migration")
        results = migration_tool.migrate_datasources(datasources_to_migrate)
        
        # Generate and display report
        report = migration_tool.generate_report(results)
        
        logger.info("Migration completed!")
        logger.info(f"Total datasources: {report['total_datasources']}")
        logger.info(f"Successful: {report['successful_migrations']}")
        logger.info(f"Failed: {report['failed_migrations']}")
        logger.info(f"Success rate: {report['success_rate']:.1f}%")
        
        if report['failed_datasources']:
            logger.info("Failed datasources:")
            for failed in report['failed_datasources']:
                logger.info(f"  - {failed['name']}: {failed['error']}")
        
        # Save report to file
        with open('migration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()