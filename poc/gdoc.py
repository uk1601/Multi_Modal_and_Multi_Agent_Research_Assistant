from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import markdown
import os
import pickle
from typing import Optional, Dict, List

class GoogleDocsConverter:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/documents']
        self.creds = self.get_credentials()
        self.service = build('docs', 'v1', credentials=self.creds)
        self.current_index = 1  # Track current position in document

    def get_credentials(self) -> Credentials:
        """Get or refresh credentials for Google Docs API"""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return creds

    def create_empty_document(self, title: str) -> str:
        """Create a new Google Doc"""
        try:
            document = self.service.documents().create(body={'title': title}).execute()
            return document.get('documentId')
        except Exception as e:
            print(f"Error creating document: {e}")
            raise

    def create_metadata_requests(self, metadata: Dict) -> List[Dict]:
        """Create requests for metadata section"""
        requests = []
        
        # Add each metadata field
        metadata_text = ""
        for key, value in metadata.items():
            if key != 'title':  # Title will be handled separately
                metadata_text += f"{key}: {value}\n"
        
        if metadata_text:
            requests.extend([
                {
                    'insertText': {
                        'location': {'index': self.current_index},
                        'text': metadata_text + "\n"
                    }
                },
                {
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': self.current_index,
                            'endIndex': self.current_index + len(metadata_text) + 1
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'NORMAL_TEXT',
                            'spaceAbove': {'magnitude': 0, 'unit': 'PT'},
                            'spaceBelow': {'magnitude': 0, 'unit': 'PT'}
                        },
                        'fields': 'namedStyleType,spaceAbove,spaceBelow'
                    }
                }
            ])
            self.current_index += len(metadata_text) + 1

        return requests
    
    def create_title_requests(self, title: str) -> List[Dict]:
        """Create requests for document title with Heading 1"""
        requests = [
            {
                'insertText': {
                    'location': {'index': self.current_index},
                    'text': title + '\n\n'
                }
            },
            {
                'updateParagraphStyle': {
                    'range': {
                        'startIndex': self.current_index,
                        'endIndex': self.current_index + len(title) + 2
                    },
                    'paragraphStyle': {
                        'namedStyleType': 'HEADING_1',
                        'alignment': 'CENTER',
                        'spaceAbove': {'magnitude': 20, 'unit': 'PT'},
                        'spaceBelow': {'magnitude': 20, 'unit': 'PT'}
                    },
                    'fields': 'namedStyleType,alignment,spaceAbove,spaceBelow'
                }
            },
            {
                'updateTextStyle': {
                    'range': {
                        'startIndex': self.current_index,
                        'endIndex': self.current_index + len(title)
                    },
                    'textStyle': {
                        'fontSize': {'magnitude': 26, 'unit': 'PT'},
                        'weightedFontFamily': {'fontFamily': 'Arial'},
                        'bold': True
                    },
                    'fields': 'fontSize,weightedFontFamily,bold'
                }
            }
        ]
        self.current_index += len(title) + 2
        return requests

    def create_section_requests(self, section: str, metadata: dict) -> List[Dict]:
        """Create requests for a content section"""
        requests = []
        lines = section.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Handle headers
            if line.startswith('#'):
                level = len(line.split()[0])  # Count # symbols
                text = line.lstrip('#').strip()
                
                # Special handling for main title (single #)
                if text.strip() == metadata.get('title', '').strip():  # This is the main title
                    requests.extend([
                        {
                            'insertText': {
                                'location': {'index': self.current_index},
                                'text': text + '\n\n'
                            }
                        },
                        {
                            'updateParagraphStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text) + 2
                                },
                                'paragraphStyle': {
                                    'namedStyleType': 'HEADING_1',
                                    'alignment': 'CENTER',
                                    'spaceAbove': {'magnitude': 20, 'unit': 'PT'},
                                    'spaceBelow': {'magnitude': 10, 'unit': 'PT'}
                                },
                                'fields': 'namedStyleType,alignment,spaceAbove,spaceBelow'
                            }
                        },
                        {
                            'updateTextStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text)
                                },
                                'textStyle': {
                                    'fontSize': {'magnitude': 26, 'unit': 'PT'},
                                    'bold': True
                                },
                                'fields': 'fontSize,bold'
                            }
                        }
                    ])
                    self.current_index += len(text) + 2
                    
                # Handle Overview, Executive Summary, and other main sections (## level)
                elif level == 2:
                    requests.extend([
                        {
                            'insertText': {
                                'location': {'index': self.current_index},
                                'text': text + '\n\n'
                            }
                        },
                        {
                            'updateParagraphStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text) + 2
                                },
                                'paragraphStyle': {
                                    'namedStyleType': 'HEADING_2',
                                    'spaceAbove': {'magnitude': 20, 'unit': 'PT'},
                                    'spaceBelow': {'magnitude': 10, 'unit': 'PT'}
                                },
                                'fields': 'namedStyleType,spaceAbove,spaceBelow'
                            }
                        },
                        {
                            'updateTextStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text)
                                },
                                'textStyle': {
                                    'fontSize': {'magnitude': 20, 'unit': 'PT'},
                                    'bold': True
                                },
                                'fields': 'fontSize,bold'
                            }
                        }
                    ])
                    self.current_index += len(text) + 2
                    
                # Handle subsections (### level)
                elif level == 3:
                    requests.extend([
                        {
                            'insertText': {
                                'location': {'index': self.current_index},
                                'text': text + '\n\n'
                            }
                        },
                        {
                            'updateParagraphStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text) + 2
                                },
                                'paragraphStyle': {
                                    'namedStyleType': 'HEADING_3',
                                    'spaceAbove': {'magnitude': 16, 'unit': 'PT'},
                                    'spaceBelow': {'magnitude': 8, 'unit': 'PT'}
                                },
                                'fields': 'namedStyleType,spaceAbove,spaceBelow'
                            }
                        },
                        {
                            'updateTextStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text)
                                },
                                'textStyle': {
                                    'fontSize': {'magnitude': 16, 'unit': 'PT'},
                                    'bold': True
                                },
                                'fields': 'fontSize,bold'
                            }
                        }
                    ])
                    self.current_index += len(text) + 2
                
                # Handle any deeper levels
                else:
                    header_level = min(level, 6)  # Max header level is 6
                    requests.extend([
                        {
                            'insertText': {
                                'location': {'index': self.current_index},
                                'text': text + '\n\n'
                            }
                        },
                        {
                            'updateParagraphStyle': {
                                'range': {
                                    'startIndex': self.current_index,
                                    'endIndex': self.current_index + len(text) + 2
                                },
                                'paragraphStyle': {
                                    'namedStyleType': f'HEADING_{header_level}',
                                    'spaceAbove': {'magnitude': 14, 'unit': 'PT'},
                                    'spaceBelow': {'magnitude': 6, 'unit': 'PT'}
                                },
                                'fields': 'namedStyleType,spaceAbove,spaceBelow'
                            }
                        }
                    ])
                    self.current_index += len(text) + 2
            
            # Handle duration lines
            elif line.startswith('Duration:'):
                requests.extend([
                    {
                        'insertText': {
                            'location': {'index': self.current_index},
                            'text': line + '\n\n'
                        }
                    },
                    {
                        'updateParagraphStyle': {
                            'range': {
                                'startIndex': self.current_index,
                                'endIndex': self.current_index + len(line) + 2
                            },
                            'paragraphStyle': {
                                'namedStyleType': 'NORMAL_TEXT',
                                'indentStart': {'magnitude': 20, 'unit': 'PT'},
                                'spaceAbove': {'magnitude': 8, 'unit': 'PT'},
                                'spaceBelow': {'magnitude': 8, 'unit': 'PT'}
                            },
                            'fields': 'namedStyleType,indentStart,spaceAbove,spaceBelow'
                        }
                    },
                    {
                        'updateTextStyle': {
                            'range': {
                                'startIndex': self.current_index,
                                'endIndex': self.current_index + len(line)
                            },
                            'textStyle': {
                                'italic': True
                            },
                            'fields': 'italic'
                        }
                    }
                ])
                self.current_index += len(line) + 2
            
            # Handle regular text
            else:
                requests.extend([
                    {
                        'insertText': {
                            'location': {'index': self.current_index},
                            'text': line + '\n'
                        }
                    },
                    {
                        'updateParagraphStyle': {
                            'range': {
                                'startIndex': self.current_index,
                                'endIndex': self.current_index + len(line) + 1
                            },
                            'paragraphStyle': {
                                'namedStyleType': 'NORMAL_TEXT',
                                'spaceAbove': {'magnitude': 4, 'unit': 'PT'},
                                'spaceBelow': {'magnitude': 4, 'unit': 'PT'}
                            },
                            'fields': 'namedStyleType,spaceAbove,spaceBelow'
                        }
                    }
                ])
                self.current_index += len(line) + 1

        return requests

    def convert_markdown_to_gdoc(self, markdown_file: str) -> Optional[str]:
        """Convert markdown file to Google Doc"""
        try:
            # Read markdown content
            with open(markdown_file, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split into metadata and sections
            sections = content.split('<!-- ------------------------ -->')
            
            # Parse metadata
            metadata_lines = sections[0].strip().split('\n')
            metadata = {}
            for line in metadata_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

            # Create document
            doc_id = self.create_empty_document(metadata.get('title', 'Research Document'))
            
            # Reset current index
            self.current_index = 1
            
            # Create requests
            requests = []
            
            # Add metadata
            requests.extend(self.create_metadata_requests(metadata))
            
            # Add title as Heading 1
            title = metadata.get('title', 'Research Document')
            requests.extend(self.create_title_requests(title))
            
            # Add sections
            for section in sections[1:]:
                if section.strip():
                    requests.extend(self.create_section_requests(section,metadata))

            # Execute requests
            if requests:
                self.service.documents().batchUpdate(
                    documentId=doc_id,
                    body={'requests': requests}
                ).execute()

            return f"https://docs.google.com/document/d/{doc_id}/edit"

        except Exception as e:
            print(f"Error converting markdown to Google Doc: {e}")
            return None

def main():
    converter = GoogleDocsConverter()
    markdown_file = "data/parsed/Beyond Active and Passive Investing_ The Customization of Finance.md"
    doc_url = converter.convert_markdown_to_gdoc(markdown_file)
    if doc_url:
        print(f"Successfully created Google Doc: {doc_url}")

if __name__ == "__main__":
    main()
