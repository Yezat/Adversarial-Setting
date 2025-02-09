import tomllib

def get_project_version():
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
    
    return pyproject_data["project"]["version"]

if __name__ == "__main__":
    print(f"Project Version: {get_project_version()}")
