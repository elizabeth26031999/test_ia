import os

def print_directory_structure(root_dir, indent=''):
    """Imprime la estructura de directorios de manera recursiva."""
    items = os.listdir(root_dir)
    for item in sorted(items):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print(f"{indent}├── {item}/")
            print_directory_structure(item_path, indent + "│   ")
        else:
            print(f"{indent}├── {item}")

# Definir el directorio raíz
root_dir = '0DETECTION'

# Imprimir la estructura de directorios
print_directory_structure(root_dir)
