#TextProcessing class to process the input data
class TextProcessing:
    def __init__(self, df, block_size):
        self.corpus = " ".join(list(df.description)).split(
            " "
        )  # Taking all descriptions from each row in dataframe
        self.tokens = list(
            set(self.corpus)
        )  # Creating a list of unique tokens from the input text
        self.str_to_int = {
            word: number for number, word in enumerate(self.tokens)
        }  # String to int dictionary (tokenising)
        self.int_to_str = {
            number: word for word, number in self.str_to_int.items()
        }  # Int to string dictionary to reverse tokenization
        self.labels = {
            label: number for number, label in enumerate(df.label.unique())
        }  # Labels to numbers dictionary
        self.block_size = block_size

    # Function for encoding (tokenizing)
    def encode(self, text):
        return [self.str_to_int[word] for word in text]

    def decode(self, encoded_text):
        return " ".join(self.int_to_str.get(index, "") for index in encoded_text)

    # Tokenizing the data and placing the numerated label at the end of it and padding
    def padding_and_tokenizing(self, df):
        words = []
        for row in df.iterrows():
            curr = self.encode(row[1].description.split(" ")) + [
                self.labels[row[1].label]
            ]
            curr = [0] * (self.block_size - len(curr)) + curr
            words.append(curr)
        return words

