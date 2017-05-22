from models import *

def main():
    base_model = create_base_model()
    generator = create_generator(base_model)

if __name__ == '__main__':
    main()
