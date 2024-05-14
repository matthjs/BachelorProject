class AttributeWrapper:
    def __init__(self, obj, *attributes):
        self.obj = obj
        self.attributes = attributes

    def __getattr__(self, attr):
        if attr in self.attributes:
            return getattr(self.obj, attr)
        else:
            raise AttributeError(f"'{type(self.obj).__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr in ['obj', 'attributes']:
            super().__setattr__(attr, value)
        elif attr in self.attributes:
            setattr(self.obj, attr, value)
        else:
            raise AttributeError(f"'{type(self.obj).__name__}' object has no attribute '{attr}'")