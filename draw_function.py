import colorsys, random
import cv2
import colorsys



def random_colors(N):
	"""
	Generate random colors.
	To get visually distinct colors, 
	generaate them in HSV space then convert 
	it to RGB (Red Green Blue)
	param: N : Nomber of classes in the Frame
	"""
	random.seed(0)
	all_colors = []
	for i in range(N):
		x, y, z = random.randint(0,255), random.randint(0,255), random.randint(0,255)
		all_colors.append((x,y,z))
	return all_colors


def draw(frame, x, y, w, h, color):
    #8 line
    frame = cv2.line(frame, (x,y), (x+w, y), color, 1)
    frame = cv2.line(frame, (x,y), (x, y+h), color, 1)
    frame = cv2.line(frame, (x,y+h), (x+w, y+h), color, 1)
    frame = cv2.line(frame, (x+w,y), (x+w, y+h), color, 1)

    return frame

def text(frame, string, color, x=0, y=0):
	font = cv2.FONT_HERSHEY_COMPLEX
	s = cv2.getTextSize(string, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
	frame = cv2.rectangle(frame, (x, y-2*s[1]), (x+s[0], y), color, -1)
	frame = cv2.putText(frame, string, (x,y-int(s[1]/2)), font, 1, (0,0,0),1)

	return frame