## Attention metric of the last layer 
if self.apply_CL_sentence:
  sentence_representation = torch.mean(outputs.hidden_states[-1],axis=1)
  cos_sim = self.contrastive_model(sentence_representation,sentence_representation)
  sim_labels = torch.arange(cos_sim.size(0)).long().to(model.device)
  loss_fn_sim = nn.CrossEntropyLoss()
  loss_sim = loss_fn_sim(cos_sim, sim_labels)


if self.apply_CL_token:
  attention_metric = outputs.attentions[-1] / self.temp
  # Get the average on the dim of heads
  attention_metric = torch.mean(attention_metric,axis=1)
  sim_labels = torch.arange(attention_metric.size(-1)).long().to(model.device)
  sim_labels = sim_labels.view(1,-1).repeat(attention_metric.size(0),1)
  loss_fn_sim = nn.CrossEntropyLoss()
  loss_sim = loss_fn_sim(attention_metric,sim_labels)
 
## Using the 3-5 layers attention
if self.apply_CL_token:
  attention_metric = torch.cat(outputs.attentions[3:6],axis=0) / self.temp
  # Get the average on the dim of heads
  attention_metric = torch.mean(attention_metric,axis=1)
  sim_labels = torch.arange(attention_metric.size(-1)).long().to(model.device)
  sim_labels = sim_labels.view(1,-1).repeat(attention_metric.size(0),1)
  loss_fn_sim = nn.CrossEntropyLoss()
  loss_sim = loss_fn_sim(attention_metric,sim_labels)
