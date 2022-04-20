# The overlap label smoothing mechanism
def compute_loss(self, model, inputs, classifier=None, model_extra=None, return_outputs=False):
        """
        How the loss is computed by Trainer_self. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss_pred = 0.
        loss_sim = None
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "overlap" in inputs:
            overlap = inputs.pop("overlap")
            overlap_target = 0.5 * torch.ones_like(overlap)
        else:
            overlap = None
        labels = inputs.pop("labels")
        smoother_labels = F.one_hot(labels, num_classes=model.config.num_labels).float()
        for index in range(len(smoother_labels)):
            if smoother_labels[index,1] == 1:
                smoother_labels[index] += 0.45 * overlap[index] / (model.config.num_labels - 1)
                smoother_labels[index,1] -= 0.45 * overlap[index] * (1 + 1/(model.config.num_labels - 1))
        outputs = model(**inputs,labels=smoother_labels)
        
# if not using one_hot label ==> the distruibution of the label 0 is None for the backward operation
# this not make sense on general but work on PAWS (60.5 acc best) and does not too harmful for the original data (dev_ori : 90.5 acc)

# The second method for smoothing labels (The weight of smoother becomes higher because at the begining of training the weight is initialized)
def compute_loss(self, model, inputs, classifier=None, model_extra=None, return_outputs=False):
        """
        How the loss is computed by Trainer_self. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss_pred = 0.
        loss_sim = None
        if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
        else:
                labels = None
        if "overlap" in inputs:
                overlap = inputs.pop("overlap")
                overlap_target = 0.5 * torch.ones_like(overlap)
        else:
                overlap = None
        if model.config.problem_type == "overlap":
                labels = inputs.pop("labels")
                smoother_labels = F.one_hot(labels, num_classes=model.config.num_labels).float()
                #smoother_labes = torch.zeros_like(laels)
                for index in range(len(smoother_labels)):
                if smoother_labels[index,1] == 1:
                    smoothing = self.state.epoch / self.num_train_epochs
                    smoother_labels[index] += smoothing * 0.45 * overlap[index] / (model.config.num_labels - 1)
                    smoother_labels[index,1] -= smoothing * 0.45 * overlap[index] * (1 + 1 / (model.config.num_labels - 1))
                outputs = model(**inputs,labels=smoother_labels)
        else:
        outputs = model(**inputs)
        
# The third method for smoothing labels (The weight of smoother becomes higher because at the begining of training the weight is initialized)
def compute_loss(self, model, inputs, classifier=None, model_extra=None, return_outputs=False):
        """
        How the loss is computed by Trainer_self. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss_pred = 0.
        loss_sim = None
        if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
        else:
                labels = None
        if "overlap" in inputs:
                overlap = inputs.pop("overlap")
                overlap_target = 0.5 * torch.ones_like(overlap)
        else:
                overlap = None
        outputs = model(**inputs)
        if model.config.problem_type == "overlap":
            labels = inputs.pop("labels")
            smoothing = self.state.epoch / self.num_train_epochs
            debias_label = torch.zeros(labels.size(0))
            for index, label in enumerate(labels):
                if label == 1:
                    debias_label = overlap[index] * smoothing
                else: pass
            logits = outputs["logits"]
            log_logits = F.log_softmax(logits,dim=-1)
            loss_debias = -torch.sum(debias_label.unsqueeze(-1) * log_logits, 1)
            loss_debias = loss_debias.mean()
             
    # forth: introduire a diff_ave to increase the influence of mnli train data
def compute_loss(self, model, inputs, classifier=None, model_extra=None, return_outputs=False):
        """
        How the loss is computed by Trainer_self. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss_pred = 0.
        loss_sim = None
        loss_debias = None

        diff_ave = 0.47297789359788456 - 0.3552240478360837
        #diff_ave = 0.6682674726384413 - 0.48239152584400735
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "overlap" in inputs:
            overlap = inputs.pop("overlap")
            overlap_target = 0.5 * torch.ones_like(overlap)
        else:
            overlap = None
            
        if model.config.problem_type == "overlap":
            labels = inputs.pop("labels")
            smoother_labels = F.one_hot(labels, num_classes=model.config.num_labels).float()
            #smoother_labes = torch.zeros_like(laels)
            for index in range(len(smoother_labels)):
                if smoother_labels[index,1] == 1:
                    smoothing = self.state.epoch / self.num_train_epochs
                    overlap_weight = max(overlap[index], 0.5 + diff_ave)
                    smoother_labels[index] += smoothing * 0.5 * overlap_weight
                    smoother_labels[index,1] -= 2 * smoothing * 0.5 * overlap_weight
            outputs = model(**inputs,labels=smoother_labels)
        else:
            outputs = model(**inputs)
        
   ## we back to smoother 2 and change the evaluation metric from .max() to .sum(), which is still noted as smoother 2 since it's the same for paws

## smoothing denote the smoothing mechanism with smoothing = min((self.epoch + 1) / self.num_train_epochs),1), which is denoted as smoothing 5
## smoothing-6 是 开口向下二次函数
# smoothing-7 是0-1 sigmoid爬升，然后线性下降

