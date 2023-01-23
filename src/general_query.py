import csv


if __name__ == "__main__":

    ### Code related to the part B of the project, i.e. general query ###

    dataset_path = "../data/_queries.csv"

    print('\033[1m' + 'Rating of a query in general' + '\033[0m')

    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = header[1:]

    print('The categories of the current dataset are: ', header)
    print('Example 1: Comedy, Italy, 120, 2000, 4')
    print('Example 2: Drama,,, 1970, 5')
    print('There are 2 modalities: \n1. Entering the query directly in the prompt \n2. Reading the queries to pose from a file')
    print('Choose the modality (type 1 or 2): ')
    modality = input()

    if modality == '1':
        print('Enter a query composed of maximum 5 words, making sure that the words are separated by a comma')
        print('If you want to skip a category, use a double comma (,,)')
        query = input()
        query = query.split(',')
        query = [x.strip() for x in query]
        query = query[:5]
        print('You entered the query:', query)

    elif modality == '2':
        print('Enter the full path of the file containing the queries')
        path = input()
        file_found = False
        while not file_found:
            try:
                queries = []
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        query = row
                        query = [x.strip() for x in query]
                        query = query[:5]
                        queries.append(query)
                print('The queries to pose are: ', queries)
                file_found = True
            except:
                print('The file does not exist, please check the full path')
                print('Enter the full path of the file containing the queries')
                path = input()
    else:
        print('Invalid modality')


