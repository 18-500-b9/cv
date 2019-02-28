// Java Code to find all three angles 
// of a triangle given coordinate 
// of all three vertices 
  
import java.awt.geom.Point2D; 
import static java.lang.Math.PI; 
import static java.lang.Math.sqrt; 
import static java.lang.Math.acos; 
  
class TriangleArea 
{ 
    // returns square of distance b/w two points 
    static float lengthSquare(Point2D.Float p1, Point2D.Float p2) 
    { 
        float xDiff = p1.x- p2.x; 
        float yDiff = p1.y- p2.y; 
        return xDiff*xDiff + yDiff*yDiff; 
    } 
      
    static void printAngle(Point2D.Float A, Point2D.Float B, 
            Point2D.Float C) 
    { 
    // Square of lengths be a2, b2, c2 
    float a2 = lengthSquare(B,C); 
    float b2 = lengthSquare(A,C); 
    float c2 = lengthSquare(A,B); 
      
    // lenght of sides be a, b, c 
    float a = (float)sqrt(a2); 
    float b = (float)sqrt(b2); 
    float c = (float)sqrt(c2); 
      
    // From Cosine law 
    float alpha = (float) acos((b2 + c2 - a2)/(2*b*c)); 
    float betta = (float) acos((a2 + c2 - b2)/(2*a*c)); 
    float gamma = (float) acos((a2 + b2 - c2)/(2*a*b)); 
      
    // Converting to degree 
    alpha = (float) (alpha * 180 / PI); 
    betta = (float) (betta * 180 / PI); 
    gamma = (float) (gamma * 180 / PI); 
      
    // printing all the angles 
    System.out.println("alpha : " + alpha); 
    System.out.println("betta : " + betta); 
    System.out.println("gamma : " + gamma); 
    } 
      
    // Driver method 
    public static void main(String[] args)  
    { 
        Point2D.Float A = new Point2D.Float(2.75f,2.75f); 
        Point2D.Float B = new Point2D.Float(35.5f,16.75f); 
        Point2D.Float C = new Point2D.Float(36.75f,15.5f); 
        // Point2D.Float A = new Point2D.Float(2.82f,2.82f); 
        // Point2D.Float B = new Point2D.Float(19.13f, 10.12f); 
        // Point2D.Float C = new Point2D.Float(18.12f,11.13f); 
        printAngle(A,B,C); 
    } 
} 