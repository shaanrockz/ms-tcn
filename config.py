import yaml

class Dummy:
    def __init__(*args, **kwargs):
        pass

class config:
    def __init__(self, external_path=None):

        if external_path:
            stream = open(external_path, "r")
            docs = yaml.safe_load_all(stream)
            for doc in docs:
                for k, v in doc.items():
                    cmd = "self."+k+"=Dummy()"
                    exec(cmd)
                    # if k == "train":
                    if type(v) is dict:
                        for k1, v1 in v.items():
                            cmd = "self."+k+"." + k1 + "=" + repr(v1)
                            print(cmd)
                            exec(cmd)
                    else:
                        cmd = "self."+k+"="+repr(v)
                        print(cmd)
                        exec(cmd)
            stream.close()
        else:
            self.algo = 'no_background'

            self.dataset = Dummy()
            self.dataset.dataset = 'cross_task'
            self.dataset.split = 1

            self.training = Dummy()
            self.training.enable = True
            self.training.epoch = 50

            self.additional_loss = Dummy()
            self.additional_loss.enable = True
            self.additional_loss.background_entropy = True
            self.additional_loss.position_awareness = True

            self.evaluation = Dummy()
            self.evaluation.enable = True
            self.evaluation.predict = True
            self.evaluation.eval = True
            self.evaluation.threshold_analysis = True

            self.evaluation_option = Dummy()
            self.evaluation_option.enable = True
            self.evaluation_option.background_probability = False
            self.evaluation_option.entropy = True

            self.metrics = Dummy()
            self.metrics.MoF = True
            self.metrics.MoF_BG = True
            self.metrics.AUC = True


if __name__ == "__main__":
    cfg = config(external_path="./configs/config_separate_background.yaml")
