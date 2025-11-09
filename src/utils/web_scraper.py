import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, base_url: str):
        """Initialize the web scraper with a base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        # Add headers to mimic a browser with full header set
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        try:
            parsed_base = urlparse(self.base_url)
            parsed_url = urlparse(url)
            return parsed_url.netloc == parsed_base.netloc
        except:
            return False

    def clean_text(self, text: str) -> str:
        """Clean scraped text by removing extra whitespace and special characters."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single page."""
        try:
            response = self.session.get(url, timeout=30)  # Increased timeout
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style']):
                element.decompose()

            # Extract main content
            content = {
                'url': url,
                'title': self.clean_text(soup.title.string) if soup.title else '',
                'headings': [self.clean_text(h.text) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
                'paragraphs': [],
                'metadata': {
                    'description': self.clean_text(soup.find('meta', {'name': 'description'})['content']) 
                    if soup.find('meta', {'name': 'description'}) else ''
                }
            }

            # Extract text from all relevant elements
            main_content = []
            
            # Look for main content areas
            content_areas = soup.find_all(['main', 'article', 'div', 'section'])
            for area in content_areas:
                # Get all text elements
                for element in area.find_all(['p', 'div', 'span', 'td', 'li', 'a']):
                    text = self.clean_text(element.get_text())
                    if text and len(text.split()) > 3:  # Only keep substantial content
                        main_content.append(text)

            content['paragraphs'] = main_content

            # Extract table content
            tables = soup.find_all('table')
            table_content = []
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = ' | '.join(self.clean_text(cell.get_text()) for cell in cells if cell.get_text().strip())
                    if row_text.strip():
                        table_content.append(row_text)
            
            content['tables'] = table_content

            return content
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def scrape_site(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Scrape multiple pages from the site, starting from base_url."""
        pages_scraped = []
        urls_to_scrape = {self.base_url}
        scraped_urls = set()
        
        # Priority paths to ensure we scrape important sections first
        priority_paths = [
            'index.html',           # Homepage
            'about.html',           # About the court
            'contact.html',         # Contact information
            'latest/index.html',    # Latest updates
            'notification.html',    # Notifications
            'circulars.html',       # Circulars
            'bench.html',           # Bench information
            'causelist.html',       # Cause lists
            'display.html',         # Display board
            'recruitment.html',     # Recruitment
            'holiday.html',         # Holiday calendar
            'rules.html',           # Court rules
            'contactus.html'        # Alternative contact page
        ]
        
        # Add priority URLs to scrape queue
        for path in priority_paths:
            priority_url = urljoin(self.base_url, path)
            if self._is_valid_url(priority_url):
                urls_to_scrape.add(priority_url)

        while urls_to_scrape and len(scraped_urls) < max_pages:
            try:
                url = urls_to_scrape.pop()
                if url in scraped_urls:
                    continue

                logger.info(f"Scraping page: {url}")
                content = self.scrape_page(url)
                if content:
                    pages_scraped.append(content)
                    scraped_urls.add(url)

                    # Find new links on the page
                    response = self.session.get(url, timeout=30)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # First, collect all links
                    all_links = []
                    for link in soup.find_all('a', href=True):
                        new_url = urljoin(url, link['href'])
                        if self._is_valid_url(new_url) and new_url not in scraped_urls:
                            all_links.append(new_url)
                    
                    # Prioritize links containing important keywords
                    priority_keywords = ['about', 'contact', 'rules', 'procedure', 'notification', 
                                      'circular', 'judgment', 'order', 'case', 'recruitment']
                    
                    for link in all_links:
                        lower_link = link.lower()
                        # Add high-priority links first
                        if any(keyword in lower_link for keyword in priority_keywords):
                            urls_to_scrape.add(link)
                        # Then add remaining valid links
                        elif len(urls_to_scrape) < max_pages * 2:  # Keep a buffer of URLs
                            urls_to_scrape.add(link)
                            
                # Small delay to avoid overwhelming the server
                import time
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                continue

        return pages_scraped