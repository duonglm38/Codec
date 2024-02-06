import torch
from utils import Candidate, MODEL2BRACKET_IDS


class BracketConstraintDecodingArgument:
    def __init__(
            self,
            template_ids: torch.LongTensor,
            template_pointer: int,
            future_steps: int,
            search_mode: int,
            batch_size: int,
            bracket_stack: list[str],
            model_name: str,
            left_marker: str = '[',
            right_marker: str = ']',
            n_best: int = 5,
            possible_opening_positions: set[int] = None,
            possible_closing_positions: dict = None,
            lb_text_ids: torch.LongTensor = None,
            lb_score: float = None,
            accumulate_scores: torch.FloatTensor = None,
            save_visualization: bool = False
    ):
        r"""

            Object to store arguments for the search function

            Parameters:
                template_ids (`torch.LongTensor`):
                    The token ids of the template to insert markers
                template_pointer (`int`):
                    The next token from the template to consider, normally is initialized with 1
                future_steps (`int`):
                    The hyperparameter of the heuristic lowerbound (i.e., alpha in the paper)
                search_mode (`int`):
                    mode = 0: fast Codec
                    mode = 1: slow Codec
                    mode = 2: Beam search in constrained space
                batch_size (`int`):
                    Number of search branches at a time
                bracket_stack (`list`):
                    Markers not inserted, in a stack order, e.g., (`]`, `[`) => insert order: `[` -> `]`
                model_name (`str`):
                    Name of the MT model to decode (e.g., `nllb`, `mbart`, `m2m`)
                left_marker (`str`):
                    Symbol of the left marker to insert
                right_marker (`str`):
                    Symbol of the right marker to insert
                n_best (`int`):
                    Number of top candidates to search
                possible_opening_positions (`set`):
                    Set of possible positions to insert the opening marker
                possible_closing_positions (`set`):
                    Set of possible positions to insert the closing marker (not used)
                lb_text_ids (`torch.LongTensor`):
                    Not used
                lb_score (`float`):
                    Not used
                accumulate_scores (`torch.FloatTensor`):
                    Not used
                save_visualization (`bool`):
                    Where the to save the search tree
            """
        self.arguments = dict()
        bracket_mapping = MODEL2BRACKET_IDS[model_name]
        stack_size = len(bracket_stack)
        if lb_score is None:
            lb_score = torch.FloatTensor([[-float("inf")]])
        else:
            lb_score = torch.FloatTensor([[lb_score]])
        self.arguments['candidate'] = Candidate(text_ids=lb_text_ids,
                                                score=lb_score, accumulate_scores=accumulate_scores
                                                )
        self.arguments['bracket_stack'] = bracket_stack
        self.arguments['bracket_mapping'] = bracket_mapping
        self.arguments['template_pointer'] = torch.LongTensor([template_pointer])
        self.arguments['stack_pointer'] = torch.LongTensor([stack_size - 1])
        self.arguments['template_ids'] = template_ids
        self.arguments['batch_size'] = batch_size
        self.arguments['future_steps'] = future_steps
        self.arguments['search_mode'] = search_mode
        self.arguments['save_visualization'] = save_visualization
        self.arguments['possible_opening_positions'] = possible_opening_positions
        self.arguments['possible_closing_positions'] = possible_closing_positions
        self.arguments['n_best'] = n_best
        self.arguments['left_marker'] = left_marker
        self.arguments['right_marker'] = right_marker

