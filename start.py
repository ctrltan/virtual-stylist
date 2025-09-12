import sys
from outfit_assembler import Assembler
from csv_creator import CSVCreator

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == 'create_outfit_csv':
            creator = CSVCreator()
            creator.create_outfit_csv()
        elif mode == 'create_clothing_csv':
            creator = CSVCreator()
            creator.create_clothing_item_csv()
        else:
            print(f"'{mode}': is not a valid mode")
            sys.exit()
    else:
        assembler = Assembler()


