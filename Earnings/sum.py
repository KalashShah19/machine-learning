import re

def extract_and_add_numbers(text):
    # Find all numbers in the text using regular expressions
    numbers = re.findall(r'\d+', text)
    # Convert the numbers from strings to integers
    numbers = list(map(int, numbers))
    # Return the sum of the numbers
    return sum(numbers)

def get_multiline_input():
    print("Enter your text (type 'END' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)

# Get multiline input from the user
multiline_text = get_multiline_input()
# Calculate the sum of numbers in the text
result = extract_and_add_numbers(multiline_text)
print(f"The sum of numbers in the text is: {result}")
