class MaskBinarization():
    def __init__(self):
        self.thresholds = 0.5
    def transform(self, predicted):
        yield predicted > self.thresholds

class TripletMaskBinarization(MaskBinarization):
    def __init__(self, triplets, with_channels=True):
        super().__init__()
        self.thresholds = triplets
        self.dims = (2,3) if with_channels else (1,2)
    def transform(self, predicted):
        for top_score_threshold, area_threshold, bottom_score_threshold in self.thresholds:
            clf_mask = predicted > top_score_threshold
            pred_mask = predicted > bottom_score_threshold
            pred_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
            yield pred_mask