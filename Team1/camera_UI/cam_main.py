import tkinter
from tkinter import StringVar
import cv2
import PIL.Image, PIL.ImageTk
import time
from invoice_compare import invoice_compare

#import pytesseract
from infer_app import infer_app#in current path
from keras_retinanet.utils.image import read_image_bgr

import threading as td

import os
import time
import numpy as np
from EAST import *
import imutils
from imutils.object_detection import non_max_suppression


video_source = 0

class App:
	def __init__(self, window, window_title, video_source=video_source):
		self.window = window
		self.window.title(window_title)
		self.video_source = video_source

		# open video source (by default this will try to open the computer webcam)
		self.vid = MyVideoCapture(self.video_source)
		
		#initial recognized number and result
		global detect_num, winning_num_set, month
		detect_num = StringVar()
		winning_num_set = StringVar()
		winning_month = StringVar()
		winning_month_info = StringVar()
		
		detect_num.set('Starting ...')#initial valeue
		winning_num_set.set('Fadatsai!!')
		
		month = 0#The newest
		#Load invoice_compare()
		global compare_mod, retina_mod
		compare_mod = invoice_compare()
		retina_mod = infer_app()
		#monthly_winning_table = compare_mod.prize_num_ls
		#upload_img_result = upload_img()#Detect number put hereeeeeeeeeeeeeeeeeeeeeeee
		#print(self.upload_img())
		
		temp = compare_mod.get_month_info(month)
		temp0 = compare_mod.get_month_info(month)[0]
		temp1 = compare_mod.get_month_info(month)[1]
		temp2 = compare_mod.get_month_info(month)[2]
		temp3 = compare_mod.get_month_info(month)[3]


		#detect_num.set(compare_mod.num_filter(self.upload_img()))#Detect number put hereeeeeeeeeeeeeeeeeeeeeeee
		winning_num_set.set(compare_mod.compare(detect_num.get(),month))
		winning_month.set(compare_mod.get_table_month(month))

		# winning numbers
		temp = self.List2String(temp)
		winning_month_info.set(temp)


		#winning_month_info.set(compare_mod.get_month_info(month))
		print("detect_num: ", detect_num.get())
		print("winning_num_set: ", winning_num_set.get())        
		print("winning_month: ", winning_month.get())
		print("winning_month_info: ", winning_month_info.get())

		# Create a canvas that can fit the above video source size
		if video_source == 1 :		
			cam_scale = 0.7
			self.canvas = tkinter.Canvas(window, width = self.vid.height*cam_scale, height = self.vid.width*(cam_scale-0.15))
			self.canvas.pack()
		else:
			self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
			self.canvas.pack()			
				
		#Label that show detect number
		self.detect_result = tkinter.Label(window, textvariable=detect_num, font=('Arial',12),width = 28, height = 2)
		self.detect_result.pack()
		#Label that show winning result
		self.winning_result = tkinter.Label(window, textvariable=winning_num_set, font=('Arial',12),width = 28, height = 2)
		self.winning_result.pack()
		#Label that show winning month

		self.frm_month_info = tkinter.Frame(window)
		self.frm_month_info.pack()

		self.frm_month_info_l = tkinter.Frame(self.frm_month_info)
		self.frm_month_info_r = tkinter.Frame(self.frm_month_info)
		self.frm_month_info_l.pack(side='left',padx=20)
		self.frm_month_info_r.pack(side='right')


		self.winning_month_show = tkinter.Label(self.frm_month_info_l,textvariable=winning_month, font=('Arial',12),width = 10, height = 2)
		self.winning_month_info_show = tkinter.Message(self.frm_month_info_r,textvariable=winning_month_info)
		self.winning_month_info_show.config(justify='left')
		self.winning_month_show.pack()
		self.winning_month_info_show.pack()
		######################################################################
		self.frm = tkinter.Frame(window)
		self.frm.pack()
		self.frm_l = tkinter.Frame(self.frm)
		self.frm_r = tkinter.Frame(self.frm)
		self.frm_l.pack(side='left',padx=20)        
		self.frm_r.pack(side='right',padx=10)

		def apply_selection():
			global compare_mod, month_list_items_set
			value = lb.get(lb.curselection())
			month_index = month_list_items_set.index(value)
			print("value: ", value)
			temp = compare_mod.get_month_info(month_index)
			temp = self.List2String(temp)
			winning_month_info.set(temp)
			winning_month.set(value)

		self.month_bt = tkinter.Button(self.frm_r, text='Apply selection',command=apply_selection, width=15, height=2)
		self.month_bt.pack()

		month_list = StringVar()
		lb = tkinter.Listbox(self.frm_l, listvariable=month_list, width=20, height=7)

		global month_list_items_set
		month_list_items_set = []
		for item in range(6):
			month_list_items = compare_mod.get_table_month(item)
			lb.insert('end',month_list_items)
			month_list_items_set.append(month_list_items)
		lb.pack()


		# After it is called once, the update method will be automatically called every delay milliseconds
		global count, semaphore
		semaphore = 0
		count = 1
		self.delay = 15
		self.update()
		self.window.mainloop()

	def List2String(self, temp):
		temp = ["特別獎：\t", temp[0].strip(), "\n特獎：\t", temp[1].strip(), "\n頭獎：\t", temp[2][0].strip(), "\n\t", temp[2][1].strip(), "\n\t", temp[2][2].strip(), "\n增開六獎：", temp[3][0].strip(), "\n\t", temp[3][1].strip()]      
		temp = "".join(temp)
		return temp


	def snapshot(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_frame()

		if ret:
			cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	
	def upload_img(self):
		#reture the frame to backednd
		global compare_mod, detect_num, winning_num_set, month, semaphore,t1     
		ret, frame = self.vid.get_frame()

		#frame = imutils.rotate(frame, 90)
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		global retina_mod, classicifer_model, EAST

		origin = frame.copy()
		H, W, D = frame.shape
		#print(H, W, D)

		Resize = (480, 480)
		rH = H / Resize[0]
		rW = W / Resize[1]
		frame = cv2.resize(frame, Resize)
		newH, newW, newD = frame.shape
		#print(newH, newW, newD)

		#blob = cv2.dnn.blobFromImage(frame, 1.0, (newH,newW), (123.68, 116.78, 103.94), swapRB=True, crop=False)
		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
		EAST.setInput(blob)
		(scores, geometry) = EAST.forward(layerNames)
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		if len(boxes) <= 0:
			detect_num.set("No Invoice Number Found")
			semaphore = 0
			return

		with classicifer_model.classifier_graph.as_default():
			#print("boxes SHAPE:",boxes.shape)
			target_box = []
			FoundBox = False
			#print("FoundBox: ",FoundBox)
			#print("boxes len: ",len(boxes))
			#print("boxes : ",boxes)
			predict_prob_max = -1
			predict_prob_max_idx = -1
			for i in range(len(boxes)):
				box = boxes[i]
				xmin, ymin, xmax, ymax = box
				Height_, Width_ = ymax - ymin, xmax - xmin
				OffsetWidth = int(Width_ * 0.3)
				OffsetHeight = int(Height_ * 0.4)
				xmin -= OffsetWidth
				xmin = max(xmin, 0)
				xmax += OffsetWidth
				xmax = min(xmax, 639)
				ymin -= OffsetHeight
				ymin = max(ymin, 0)
				ymax += OffsetHeight
				ymax = min(ymax, 439)
				#crop_img = frame[ymin:ymax,xmin:xmax,:]
				
				xmin = int(xmin * rW)
				xmax = int(xmax * rW)
				ymin = int(ymin * rH)
				ymax = int(ymax * rH)
				try:
					crop_img = origin[ymin:ymax,xmin:xmax,:]
					crop_resize_img = cv2.resize(crop_img,(250,250))/255
				except:
					print("Box is broken")
					continue
				crop_resize_img = np.expand_dims(crop_resize_img, axis=0)       

				result = classicifer_model.model.predict(crop_resize_img)#....
				probability_to_label = np.argmax(result)

				if result[0][0] > predict_prob_max :
					predict_prob_max = result[0][0]
					predict_prob_max_idx = i
				#print("Prediction:", probability_to_label, result)
				#print("Cur image size:", xmax - xmin, ymax - ymin)
				#print("NO FoundBox QQQQQQQQQQQQQQQQQQQQ")
				if probability_to_label == 0:
					target_box = box
					FoundBox = True
					print("I found the FoundBox !!!!!!!!!!!!!!")
					break          

			if not FoundBox:
				target_box = boxes[predict_prob_max_idx]
			xmin, ymin, xmax, ymax = target_box
			Height_, Width_ = ymax - ymin, xmax - xmin
			OffsetWidth = int(Width_ * 0.3)
			OffsetHeight = int(Height_ * 0.4)
			xmin -= OffsetWidth
			xmin = max(xmin, 0)
			xmax += OffsetWidth
			xmax = min(xmax, 639)
			ymin -= OffsetHeight
			ymin = max(ymin, 0)
			ymax += OffsetHeight
			ymax = min(ymax, 439)
			#crop_img = frame[ymin:ymax,xmin:xmax,:]
			
			xmin = int(xmin * rW)
			xmax = int(xmax * rW)
			ymin = int(ymin * rH)
			ymax = int(ymax * rH)
			crop_img = origin[ymin:ymax,xmin:xmax,:]

		with retina_mod.graph.as_default():
			'''
			crop_img[:,:,0] = crop_img[:,:,0] - 103.939
			crop_img[:,:,1] = crop_img[:,:,1] - 116.779
			crop_img[:,:,2] = crop_img[:,:,2] - 123.68
			'''
			# print("crop_img shape:",crop_img.shape)
			# cv2.imshow("Crop", crop_img)
			# cv2.waitKey(0)
			# cv2.destroyWindow("Crop")
			text = retina_mod.pred_string(crop_img)
			detect_num.set(text)#Detect number put hereeeeeeeeeeeeeeeeeeeeeeee
			winning_num_set.set(compare_mod.compare(detect_num.get(),month))
			print("Predict result",text)
			print("winning_num_set",winning_num_set.get())
		semaphore = 0
		#print("semaphore in upload_img: ", semaphore)
			#print("text :", text)

	def update(self):
		# Get a frame from the video source
		global count, out_image, semaphore
		ret, frame = self.vid.get_frame()
		#frame = imutils.rotate(frame, 90)

		if ret:
			mask_blur = np.ones((frame.shape))*0.6
			mask_blur[80:80+100, 40:40+240,:] = 1
			masked = mask_blur * frame
			masked_line = cv2.line(masked, (40,80), (40+240,80), (0, 0, 255), 2)
			masked_line = cv2.line(masked_line, (40,80), (40,80+100), (0, 0, 255), 2)
			masked_line = cv2.line(masked_line, (40,80+100), (40+240,80+100), (0, 0, 255), 2)
			masked_line = cv2.line(masked_line, (40+240,80), (40+240,80+100), (0, 0, 255), 2)

			self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
			self.canvas.create_image(10, 10, image = self.photo, anchor = tkinter.NW)            
			count += 1
			if (count > 20) and (semaphore == 0):#
				global compare_mod, detect_num, winning_num_set, month
				count = 0
				semaphore = 1
				t1 = td.Thread(target = self.upload_img)
				t1.start()
		
		self.window.after(self.delay, self.update)


class MyVideoCapture:
	def __init__(self, video_source=video_source):
		# Open the video source
		self.vid = cv2.VideoCapture(video_source)
		self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)#160*120
		self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

		if not self.vid.isOpened():
			raise ValueError("Unable to open video source", video_source)

		# Get video source width and height
		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print("self.width: ", self.width)
		print("self.height: ", self.height)
	def get_frame(self):
		if self.vid.isOpened():
			ret, frame = self.vid.read()
			if video_source == 1:
				frame = imutils.rotate_bound(frame, 90)
			if ret:
				# Return a boolean success flag and the current frame converted to BGR
				return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else:
				return (ret, None)
		else:
			return (ret, None)

	# Release the video source when the object is destroyed
	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()

#When open the app, load retina model first
global retina_mod, classicifer_model, EAST
retina_mod = infer_app()
classicifer_model = Classifier()

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
EASTPATH = "model/frozen_east_text_detection.pb"
print("[INFO] Loading the pre-trained EAST text detector ...")
EAST = cv2.dnn.readNet(EASTPATH)
print("[INFO] Loading the pre-trained EAST text completed")

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Invoice lottery (Tkinter and OpenCV support)")#detect_input