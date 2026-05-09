## Project Details 

The key goal of this project and repository is to investigate the robustness of existing black-box LLM fingerprinting and auditing techniques. 

Particularly, this experiment is broken down into 2 parts:

### Fingerprint Robustness

We intend to show that existing black-box intrinsic LLM fingerprinting techniques are TOO robust in model space (explore the knowledge base for more information), i.e., they fingerprint too many fine-tuned derivatives of an existing model as the base model. To investigate this, we:
1. Take a lineage of models (currently starting with the OLMo2-0425-1B family)
2. Construct the fingerprints for each major model in the family
3. Treating each model as the 'reference', verify the fingerprint against the entire lineage as the 'models of interest', to see if the technique fingerprints the models as being the same as the reference. E.g., taking the SFT variant as the 'reference', we verify whether each model in the lineage is fingerprinted as being the same as the SFT model. 
4. Quantifying and visualising these results to understand the extent to which a model needs to be fine-tuned, and using what techniques, to result in a different fingerprint arising 

We will repeat this for different model lineages, and potentially model distillations too (E.g., comparing fingerprints of OLMo2-7B against 1B models to see if they're identified as derivatives) + analyse the results. 

**End Goal:** Using the results of this comparison, we aim to be able to pinpoint a rough "fine-tuning radius" within which any model is seen as being a 'derivative of the original' and leads to it having the same fingerprint. 

### Audit Robustness 

We then separately attempt to show that current fairness audit sets for LLM are TOO robust in parameter space. I.e., we show that it is possible for an evasive model owner to carry out a targeted fine-tune of a model wherein they:
1. Preserve the fairness/score/performance of the model on the fairness audit set (chosen as HolisticBias)
2. Degrade the fairness of the model elsewhere (using Anthropic's harmless-base DPO set with inverted DPO), generally fairwashing it. 
Moreover, to keep the situation realistic and ensure that the model's general performance is not degraded, we try to retain its performance on other benchmarks for mathematical reasoning, CoT, general conversational ability etc.

Moreover, with the radius defined above, it gives us an upper-bound on the number of data points we want to use for the fine-tuning such that we're able to degrade performance while STILL being fingerprinted as the base model. 
