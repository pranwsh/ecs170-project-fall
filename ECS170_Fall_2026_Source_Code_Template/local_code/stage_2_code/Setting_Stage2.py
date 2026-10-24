# Experiment setting for Stage 2: use the provided train/test files.
# Clean this up later if we change the validation strategy, but it works for now.

import numpy as np
from sklearn.model_selection import train_test_split

from local_code.base_class.setting import setting


def to_numpy(values):
    # Just making sure whatever we get turns into a numpy array safely
    # Sometimes torch tensors act weirdly so gotta check for detach
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


class Setting_Stage2(setting):
    """Train on train.csv and evaluate on test.csv.

    The data is already split, so there is no K-fold or random train/test split
    like in Stage 1. For experiment comparisons, set use_validation = True to
    hold out part of train.csv as a validation set and leave test.csv untouched
    until the final run.
    """

    test_dataset = None
    use_validation = False
    validation_size = 0.2
    seed = 2

    # filled in by load_run_save_evaluate so the script can use them afterwards
    eval_data = None
    learned_result = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestDataset=None):
        # sDataset loads train.csv, sTestDataset loads test.csv
        super().prepare(sDataset, sMethod, sResult, sEvaluate)
        self.test_dataset = sTestDataset

    def print_setup_summary(self):
        super().print_setup_summary()
        if self.use_validation:
            print(f'evaluating on {self.validation_size:.0%} validation split of train.csv (test.csv not used)')
        else:
            print('evaluating on test.csv')

    def load_run_save_evaluate(self):
        # load dataset - let's grab train first
        train = self.dataset.load()

        if self.use_validation:
            # same stratified 80/20 split as Daniel's preprocessing experiments (seed 2)
            X_train, X_eval, y_train, y_eval = train_test_split(
                train['X'], train['y'], test_size=self.validation_size,
                random_state=self.seed, stratify=train['y'])
        else:
            if self.test_dataset is None:
                raise ValueError('pass the test.csv loader to prepare() or set use_validation = True')
            test = self.test_dataset.load()
            X_train, y_train = train['X'], train['y']
            X_eval, y_eval = test['X'], test['y']

        print(f'train: {len(y_train)} samples, eval: {len(y_eval)} samples')
        self.eval_data = {'X': X_eval, 'y': y_eval}

        # run MethodModule ('test' is the key the template's Method_MLP expects)
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_eval, 'y': y_eval}}
        if self.use_validation:
            # Method_MLP also tracks val loss/accuracy each epoch when data 'val' is given.
            #this was basically  only done for the validation split so test.csv never influences training choices.
            self.method.data['val'] = {'X': X_eval, 'y': y_eval}
        learned_result = self.method.run()

        # make predictions plain NumPy arrays so they can be pickled and passed to sklearn
        learned_result['pred_y'] = to_numpy(learned_result['pred_y'])
        learned_result['true_y'] = to_numpy(learned_result['true_y'])

        # perepoch train/val loss and accuracy from Method_MLP, used for the convergence plots
        history = getattr(self.method, 'history', None) or {}

        # Doing a simple loop here instead of dict comprehension just in case it errors out
        processed_history = {}
        for key, values in history.items():
            if values:
                processed_history[key] = [float(v) for v in values]

        learned_result['history'] = processed_history
        self.learned_result = learned_result

        #saving info hhere 
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
