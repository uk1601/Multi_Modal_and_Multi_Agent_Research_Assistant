import os
import re
import subprocess
import sys
from datetime import datetime


def convert_image_urls_to_links(markdown_content: str) -> str:
    """Convert image URLs to clickable links in markdown content."""
    # Pattern to match markdown image syntax: ![alt text](url)
    image_pattern = r'!\[(.*?)\]\((https?://[^\s\)]+)\)'

    def replace_with_link(match):
        alt_text = match.group(1)
        url = match.group(2)
        # Convert image to a clickable link with the original alt text
        return f'[🖼️ {alt_text or "View Image"}]({url})'

    # Replace all image patterns with links
    return re.sub(image_pattern, replace_with_link, markdown_content)


def sanitize_markdown(markdown_content: str) -> str:
    """Sanitize markdown content by converting image URLs to links."""
    # First handle image URLs
    content = convert_image_urls_to_links(markdown_content)

    # Ensure all external links open in new tab by adding {target="_blank"}
    link_pattern = r'\[(.*?)\]\((https?://[^\s\)]+)\)'

    def add_target_blank(match):
        text = match.group(1)
        url = match.group(2)
        return f'[{text}]({url})'

    content = re.sub(link_pattern, add_target_blank, content)

    return content


def export_codelab(markdown_file):
    """Export a codelab from the provided Markdown file using claat."""
    print(f"Exporting codelab from {markdown_file}...")
    try:
        # Run the claat export command
        subprocess.run(["claat", "export", markdown_file], check=True)
        print(f"Codelab exported successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting codelab: {e}")
        sys.exit(1)


def serve_codelab():
    """Serve the codelab locally."""
    print(f"Starting local codelab server...")
    try:
        # Run the claat serve command
        subprocess.run(["claat", "serve", "&","-addr", "0.0.0.0:9090"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error serving codelab: {e}")
        sys.exit(1)


def process_markdown_string(markdown_content: str):
    """Process markdown string by saving to output_markdown folder and running main function."""
    # Create output_markdown folder if it doesn't exist
    output_dir = "output_markdown"
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize the markdown content
    safe_content = sanitize_markdown(markdown_content)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"codelab_{timestamp}.md")

    print(f"Processing markdown content to file: {file_path}")

    # Write sanitized content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(safe_content)

    try:
        main(file_path)
    finally:
        # Optionally clean up the file after processing
        # os.remove(file_path)  # Uncomment if you want to clean up
        pass


def main(md_path: str):
    """Main function to process and serve the codelab."""
    # Check if the Markdown file exists
    if not os.path.isfile(md_path):
        sys.exit(f"Markdown file not found: {md_path}")

    # Export the codelab
    export_codelab(md_path)

    # Serve the codelab
    serve_codelab()


if __name__ == "__main__":
    # Example usage
    markdown_content = """
    # My Codelab

    Here's an image that will be converted to a link:
    ![Example Image](https://example.com/image.jpg)

    Here's a regular link:
    [Visit Website](https://example.com)
    """
    process_markdown_string(markdown_content)