class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        Your code should initialize the following fields:
            self.stack: The current stack represented as a list with the top of the stack as the
                        last element of the list.
            self.buffer: The current buffer represented as a list with the first item on the
                         buffer as the first item of the list
            self.dependencies: The list of dependencies produced so far. Represented as a list of
                    tuples where each tuple is of the form (head, dependent).
                    Order for this list doesn't matter.

        The root token should be represented with the string "ROOT"

        Args:
            sentence: The sentence to be parsed as a list of words.
                      Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not use it in your code.
        self.sentence = sentence

        ### START CODE HERE

        self.stack = ["ROOT"]
        
        self.buffer = list(sentence)
        
        self.dependencies = []
        
        ### END CODE HERE

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        Args:
            transition: A string that equals "S", "LA", or "RA" representing the shift, left-arc,
                        and right-arc transitions. You can assume the provided transition is a legal
                        transition.
        """
        ### START CODE HERE
        
        match transition:
            case "S":
                self.stack.append(self.buffer.pop(0))
            case "LA":
                head = self.stack[-1]
                dependent = self.stack[-2]
                self.dependencies.append((head, dependent))
                del self.stack[-2]
            case "RA":
                head = self.stack[-2]
                dependent = self.stack[-1]
                self.dependencies.append((head, dependent))
                self.stack.pop()
        
        ### END CODE HERE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        Args:
            transitions: The list of transitions in the order they should be applied
        Returns:
            dependencies: The list of dependencies produced when parsing the sentence. Represented
                          as a list of tuples where each tuple is of the form (head, dependent)
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, device, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        sentences: A list of sentences to be parsed (each sentence is a list of words)
        model: The model that makes parsing decisions. It is assumed to have a function
               model.predict(partial_parses) that takes in a list of PartialParses as input and
               returns a list of transitions predicted for each parse. That is, after calling
                   transitions = model.predict(partial_parses)
               transitions[i] will be the next transition to apply to partial_parses[i].
        device: The device to be used
        batch_size: The number of PartialParses to include in each minibatch
    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
                      Ordering should be the same as in sentences (i.e., dependencies[i] should
                      contain the parse for sentences[i]).
    """

    dependencies = None

    ### START CODE HERE
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    active_parses = partial_parses.copy()

    while active_parses:
        batch = active_parses[:batch_size]
        transitions = model.predict(batch, device)

        for parse, transition in zip(batch, transitions):
            parse.parse_step(transition)

        # Filter out completed parses
        active_parses = [p for p in active_parses if p.buffer or len(p.stack) > 1]

    return [parse.dependencies for parse in partial_parses]
    ### END CODE HERE

    return dependencies


# (Other parts of the script remain unchanged)

if __name__ == '__main__':
    test_parse_step()
    test_parse()
    test_minibatch_parse()
