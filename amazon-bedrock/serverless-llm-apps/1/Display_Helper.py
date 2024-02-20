from IPython.display import HTML, display
import html

class Display_Helper: 

    def wrap(self, long_string):
        escaped_string = html.escape(long_string)
        html_string = escaped_string.replace("\n", "<br>")
        display(HTML(f"<div style='width: 600px; word-wrap: break-word;'>{html_string}</div>"))

    def text_file(self, file_path):
        try:
            # Open the file and read its contents into a string
            with open(file_path, 'r', encoding='utf-8') as file:
                text_string = file.read()

            # Print the contents (optional)
            print(f"{file_path}:")
            self.wrap(text_string)

        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")