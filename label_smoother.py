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
             

