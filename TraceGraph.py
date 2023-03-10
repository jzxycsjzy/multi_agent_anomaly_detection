import pandas as pd

class TraceGraph:
    def __init__(self, trace_id="", span_id="", parent_span="") -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.log_info = []
        self.parent_span = parent_span
        self.child_spans = []

    def Save2Pandas(self):
        return [self.trace_id, self.span_id, self.parent_span, self.child_span, self.log_info]
    
    def LoadFromPandas(self, pd_data: pd.Series):
        self.trace_id = pd_data['traceid']
        self.span_id = pd_data['spanid']
        self.parent_span = pd_data['parentspan']
        self.child_spans = pd_data['childspan']
        self.log_info = pd_data['loginfo']

    def AddChildSpan(self, child):
        if child.parent_span == self.span_id:
            self.child_spans.append(child)


    def PrintAll(self):
        print(self.trace_id, self.span_id, self.parent_span, self.child_spans, self.log_info)