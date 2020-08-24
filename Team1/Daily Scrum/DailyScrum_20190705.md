# Daily Scrum Stand Up Meeting

Date: 2019/07/05 13:30 - 13:42

### Topic: What did you do yesterday?
- LongYing: 
	- Try finding the font for invoice, had no result.
	- Also asked friend for the font, guessed that it is 標楷體.
	- Found EMNIST dataset, including handwritten digits and digital digits.
	- Optimized the invoice number crawlerm making it faster.
- ChiaChe: 
	- Trained a YOLOv2 model on the SVHN(Street View House Numbers) dataset.
- HsuanHuai: 
	- Wrote a simple regualr expression for the digit filter.
- YuLong: 
	- Tried a single digit prediction repository, works fine.
	- Tried a multi-digit prediction, which concatenates multiple single digit image and predicts simultaneously, works fine.

### Topic: What will you do today?
- LongYing: 
	- Help with digit detection.
- ChiaChe: 
	- Keep on training the model.
- HsuanHuai: 
	- Improve digit filter.
- YuLong: 
	- Try other models. 
	- Test on our invoice data.

### Topic: Are there any impediments in your way?
- LongYing:
- ChiaChe: 
	- The SVHN dataset is too big, training takes too much time.
		- Solved: First try on a small portion of SVHN and see if it works, then train on AWS.
	- Loss becomes NaN while training
		- Might be because of the noisy data.
		- Or the learning rate might be too big.
		- Training with smaller learning rate works now, but don't know how it performs.
- HsuanHuai: 
- YuLong: 