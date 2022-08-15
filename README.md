# Extending Bias-Bench for Sentence-Transformers
---

## How to run the scripts diagnose and debias sentence-transformers.
- For benchmarking a Sentence Transformer model run ```./experiments/seat.py``` with ```sentence-transformers/all-MiniLM-L6-v2``` as the model
- Formula for ```WEAT effect size``` given by Caliskan et al is in the image below (implementation is straightforward, see code)
- Download and extract ```wikipedia-2.5.txt.zip``` in ```/data/text/``` path of bias-bench repo
- For getting debiased subspace run ```inlp_projection_matrix.py``` with specific bias types and model as ```sentence-transformers/all-MiniLM-L6-v2```. Save the subspace files. This step needs nltk.download('punkt')
- For debiasing specific bias types run ```seat_debias.py``` with respective subspace files created during previous step.
  - For gender run the following tests: ```sent-weat6 sent-weat6b sent-weat7 sent-weat7b sent-weat8 sent-weat8b``` 
  - For race run the following tests: ```sent-weat3 sent-weat3b sent-weat4 sent-weat5 sent-weat5b sent-angry_black_woman_stereotype sent-angry_black_woman_stereotype_b``` 
  - For religion run the following tests: ```sent-religion1 sent-religion1b sent-religion2 sent-religion2b```
- Optionally filter the results by significance level i.e p-value < 0.01
- Calculate average of the absolute values to get avg.effect size.
- Note: Currently only works for ```BERT or RoBERTA``` like models, the original implementation has not leveraged AutoModel implementation from HuggingFace. So if you want to test mpnet based model you need to add support for that model or include ```AutoModel``` support.


<br>
<img width="300" height= "350" alt="Fig 1" src="https://user-images.githubusercontent.com/7071019/184587449-539699cf-d404-4351-a066-e83ad9647295.png">

<br>

