import os
import subprocess
from pathlib import Path
import install_claat

class ClaatHandler:
    def __init__(self):
        # Get the absolute path of the current directory
        self.base_dir = Path.cwd().absolute()
        
        # Define paths
        self.input_md = self.base_dir / "abc/research_summary_20241115_012352.md"
        self.output_dir = self.base_dir / "codelab-dir"
        
        # Get claat path (assuming it's in the system PATH)
        self.claat = "claat"  # or full path to claat binary

    def export_codelab(self):
        """Export markdown to codelabs format"""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(exist_ok=True)
            
            # Command to export
            cmd = [
                self.claat,
                "export",
                str(self.input_md)
            ]
            
            # Run export command
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),  # Set working directory
                check=True,
                capture_output=True,
                text=True
            )
            
            print("Export output:", result.stdout)
            if result.stderr:
                print("Export errors:", result.stderr)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
            print(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
            return False

    def serve_codelab(self, port=9090):
        """Serve the codelabs"""
        try:
            # Command to serve
            cmd = [
                self.claat,
                "serve",
                "-addr", f"localhost:{port}",
                "-prefix", "/",
                str(self.output_dir)
            ]
            
            print(f"\nRunning command: {' '.join(cmd)}")
            print(f"Serving codelabs at: http://localhost:{port}/")
            
            # Run serve command
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Serve failed: {e}")
            if hasattr(e, 'output'):
                print(f"Output: {e.output}")

def main():
    installer= install_claat.ClaatInstaller()
    installer.download_claat()
    handler = ClaatHandler()
    
    # Export markdown to codelabs
    if handler.export_codelab():
        print("\nExport successful!")
        print(f"Files created in: {handler.output_dir}")
        print("\nStarting local server...")
        
        # Serve the codelabs
        handler.serve_codelab()
    else:
        print("Export failed, not starting server")

if __name__ == "__main__":
    main()