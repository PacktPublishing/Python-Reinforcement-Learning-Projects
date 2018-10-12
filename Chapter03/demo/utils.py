'''
Created on May 17, 2016

@author: a0096049
'''
import numpy
from pygame import Rect


class Color:
    
    RED   = (255, 51, 153, 220)
    GREEN = (0, 204, 0, 220)
    BLUE  = (10, 102, 240, 220)
    GRAY  = (200, 200, 200, 128)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BACKGROUND = BLACK

# Calc the gradient 'm' of a line between p1 and p2
def calculateGradient(p1, p2):
  
    # Ensure that the line is not vertical
    if (p1[0] != p2[0]):
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
        return m
    else:
        return None
 
# Calc the point 'b' where line crosses the Y axis
def calculateYAxisIntersect(p, m):
    return p[1] - (m * p[0])
 
# Calc the point where two infinitely long lines (p1 to p2 and p3 to p4) intersect.
# Handle parallel lines and vertical lines (the later has infinate 'm').
# Returns a point tuple of points like this ((x,y),...)  or None
# In non parallel cases the tuple will contain just one point.
# For parallel lines that lay on top of one another the tuple will contain
# all four points of the two lines
def getIntersectPoint(p1, p2, p3, p4):
    
    m1 = calculateGradient(p1, p2)
    m2 = calculateGradient(p3, p4)
      
    # See if the the lines are parallel
    if (m1 != m2):
        # See if either line is vertical
        if (m1 is not None and m2 is not None):
            # Neither line vertical           
            b1 = calculateYAxisIntersect(p1, m1)
            b2 = calculateYAxisIntersect(p3, m2)   
            x = (b2 - b1) / (m1 - m2)       
            y = (m1 * x) + b1           
        else:
            # Line 1 is vertical so use line 2's values
            if (m1 is None):
                b2 = calculateYAxisIntersect(p3, m2)   
                x = p1[0]
                y = (m2 * x) + b2
            # Line 2 is vertical so use line 1's values               
            elif (m2 is None):
                b1 = calculateYAxisIntersect(p1, m1)
                x = p3[0]
                y = (m1 * x) + b1           
            else:
                assert False
               
        return ((x,y),)
    else:
        # Parallel lines with same 'b' value must be the same line so they intersect
        # everywhere in this case we return the start and end points of both lines
        # the calculateIntersectPoint method will sort out which of these points
        # lays on both line segments
        b1, b2 = None, None # vertical lines have no b value
        if m1 is not None:
            b1 = calculateYAxisIntersect(p1, m1)
           
        if m2 is not None:   
            b2 = calculateYAxisIntersect(p3, m2)
       
        # If these parallel lines lay on one another   
        if b1 == b2:
            return p1,p2,p3,p4
        else:
            return None
 
# For line segments (ie not infinitely long lines) the intersect point
# may not lay on both lines.
#   
# If the point where two lines intersect is inside both line's bounding
# rectangles then the lines intersect. Returns intersect point if the line
# intesect o None if not
def calculateIntersectPoint(p1, p2, p3, p4):
  
    p = getIntersectPoint(p1, p2, p3, p4)
  
    if p is not None:               
        width = p2[0] - p1[0]
        height = p2[1] - p1[1]       
        r1 = Rect(p1, (width , height))
        r1.normalize()
       
        width = p4[0] - p3[0]
        height = p4[1] - p3[1]
        r2 = Rect(p3, (width, height))
        r2.normalize()              
    
        # Ensure both rects have a width and height of at least 'tolerance' else the
        # collidepoint check of the Rect class will fail as it doesn't include the bottom
        # and right hand side 'pixels' of the rectangle
        tolerance = 1
        if r1.width < tolerance:
            r1.width = tolerance
                    
        if r1.height < tolerance:
            r1.height = tolerance
        
        if r2.width < tolerance:
            r2.width = tolerance
                    
        if r2.height < tolerance:
            r2.height = tolerance
    
        for point in p:                 
            try:
                point = [numpy.rint(pp) for pp in point] 
                res1 = r1.collidepoint(point)
                res2 = r2.collidepoint(point)
                if res1 and res2:
                    point = [int(pp) for pp in point]                       
                    return point
            except:         
                print("point was invalid {}".format(point))
                
        # This is the case where the infinately long lines crossed but 
        # the line segments didn't
        return None            
    
    else:
        return None
        
        
# Test script below...
if __name__ == "__main__":
 
    # line 1 and 2 cross, 1 and 3 don't but would if extended, 2 and 3 are parallel
    # line 5 is horizontal, line 4 is vertical
    p1 = (1,5)
    p2 = (4,7)
    
    p3 = (4,5)
    p4 = (3,7)
    
    p5 = (4,1)
    p6 = (3,3)
    
    p7 = (3,1)
    p8 = (3,10)
    
    p9 =  (0,6)
    p10 = (5,6)
    
    p11 = (472.0, 116.0)
    p12 = (542.0, 116.0)  
    '''
    assert None != calculateIntersectPoint(p1, p2, p3, p4), "line 1 line 2 should intersect"
    assert None != calculateIntersectPoint(p3, p4, p1, p2), "line 2 line 1 should intersect"
    assert None == calculateIntersectPoint(p1, p2, p5, p6), "line 1 line 3 shouldn't intersect"
    assert None == calculateIntersectPoint(p3, p4, p5, p6), "line 2 line 3 shouldn't intersect"
    assert None != calculateIntersectPoint(p1, p2, p7, p8), "line 1 line 4 should intersect"
    assert None != calculateIntersectPoint(p7, p8, p1, p2), "line 4 line 1 should intersect"
    assert None != calculateIntersectPoint(p1, p2, p9, p10), "line 1 line 5 should intersect"
    assert None != calculateIntersectPoint(p9, p10, p1, p2), "line 5 line 1 should intersect"
    assert None != calculateIntersectPoint(p7, p8, p9, p10), "line 4 line 5 should intersect"
    assert None != calculateIntersectPoint(p9, p10, p7, p8), "line 5 line 4 should intersect"
    '''
    '''
    p13 = (106, 96)
    p14 = (212, 96)
    p15 = (139, 115)
    p16 = (100, 46)
    '''
    p13 = (139, 115)
    p14 = (100, 46)
    p15 = (106, 96)
    p16 = (212, 96)
    
    point = calculateIntersectPoint(p13, p14, p15, p16)
    assert None != point, "line 7 line 8 should intersect"
    print(point)
    
    print("\nSUCCESS! All asserts passed for doLinesIntersect")
    