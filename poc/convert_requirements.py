import toml
import sys

def convert_poetry_to_requirements():
    # Read pyproject.toml
    with open('pyproject.toml', 'r') as f:
        poetry_config = toml.load(f)
    
    # Get dependencies
    dependencies = poetry_config['tool']['poetry']['dependencies']
    dev_dependencies = poetry_config['tool']['poetry'].get('dev-dependencies', {})
    
    # Convert to pip format
    requirements = []
    
    # Process main dependencies
    for package, version in dependencies.items():
        if package == 'python':
            continue
        if isinstance(version, str):
            requirements.append(f"{package}=={version}")
        elif isinstance(version, dict):
            if 'version' in version:
                requirements.append(f"{package}=={version['version']}")
            else:
                requirements.append(package)
    
    # Write requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
        f.write('\n')

if __name__ == "__main__":
    convert_poetry_to_requirements()