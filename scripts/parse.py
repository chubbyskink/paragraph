import json

PATH = "/Users/gavinolsen/Desktop/code/data/"
STORIES = 2

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def process_json_data(json_data):

    data = json_data['data']

    stories = ""
    questionsRes = ""
    answersRes = ""

    for i in range(STORIES):
        # Extracting data from JSON
        story = data[i]['story']
        questions = data[i]['questions']
        answers = data[i]['answers']

        stories += story + "\n"
        # Writing the questions to a file
        questionsRes += ''.join([f"{q['input_text']}\n" for i, q in enumerate(questions)])

        # Writing the answers to a file
        answersRes += ''.join([f"{a['input_text']}\n" for i, a in enumerate(answers)])

    write_to_file(PATH+'questions.txt', questionsRes)
    write_to_file(PATH+'answers.txt', answersRes)
    write_to_file(PATH+'story.txt', stories)
    

if __name__ == '__main__':
    file_path = PATH+'data.json'  # Replace with your JSON file path
    json_data = read_json_file(file_path)
    process_json_data(json_data)
