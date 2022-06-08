import numpy as np;
import math;
import numpy.random as npr;
from scipy.spatial import Delaunay;
import matplotlib.pyplot as plt;

def HyperbolicPoints(n,r):
	eup = npr.random((n+1,2));
	rho = r*r/(1-r*r);
	for i in range(0,n+1):
		u = eup[i,0];
		theta = eup[i,1];
		r = np.sqrt( 1 - 1 / ( 1 + rho* u) );
		eup[i,0] = r*np.cos(2*np.pi*theta);
		eup[i,1] = r*np.sin(2*np.pi*theta);
	eup[n,0] = 0.0001;
	eup[n,1] = 0.0001;
	return eup;

def PPP(ell,r):
	mu = ell * 4 * np.pi * r / (1-r*r);
	n = npr.poisson(mu);
	points = HyperbolicPoints(n,r);
	return points;

def HCC(p1,p2,p3):
#	p3 = 1/p1.conjugate();
	a = np.linalg.det( [
		[p1.real, p1.imag, 1.0],
		[p2.real, p2.imag, 1.0],
		[p3.real, p3.imag, 1.0]])
	b = np.linalg.det( [
		[abs(p1)**2, p1.real,1.0],
		[abs(p2)**2, p2.real,1.0],
		[abs(p3)**2, p3.real,1.0]])*1.0j -np.linalg.det( [
		[abs(p1)**2, p1.imag,1.0],
		[abs(p2)**2, p2.imag,1.0],
		[abs(p3)**2, p3.imag,1.0]])
#	c = np.linalg.det( [
#		[abs(p1)**2, p1.real, p1.imag],
#		[abs(p2)**2, p2.real, p2.imag],
#		[abs(p3)**2, p3.real, p3.imag]])
#	r = np.sqrt( abs(b)**2 - 4.0*a*c) / 2.0 / abs(a)
	return -b/2.0/a;

def HCR(p1,p2,p3):
###	p3 = 1/p1.conjugate();
	a = abs(p1 - p2);
	b = abs(p2 - p3);
	c = abs(p3 - p1);
	return a*b*c/np.sqrt( ( a + b + c)*(-a+b+c)*(a-b+c)*(a+b-c) );

#def HCR(p1,p2,p3):
#	z = HCC(p1,p2,p3);
#	return abs( z - p1);

def checkFinite(p):
	p1=p[0]
	p2=p[1]
	p3=p[2]
	z = HCC(p1[0] + p1[1]*1.0j,
			p2[0] + p2[1]*1.0j,
			p3[0] + p3[1]*1.0j);
	r = p1[0] + p1[1]*1.0j - z;
	return (( abs(z) + abs(r) ) < 1);



def drawTri(p):
	p1=p[0]
	p2=p[1]
	p3=p[2]
	z1 = p1[0] + p1[1]*1.0j;
	z2 = p2[0] + p2[1]*1.0j;
	z3 = p3[0] + p3[1]*1.0j;

	r1 = HCR(z1,z2, 1/z1.conjugate());
	r2 = HCR(z3,z2, 1/z2.conjugate());
	r3 = HCR(z1,z3, 1/z3.conjugate());
	
	b1 = 0 if (p1[1] * p2[0] - p1[0] * p2[1]) < 0 else 1;
	b2 = 0 if (p2[1] * p3[0] - p2[0] * p3[1]) < 0 else 1;
	b3 = 0 if (p3[1] * p1[0] - p3[0] * p1[1]) < 0 else 1;

	print '<path d="M{},{} A{},{} 0 0,{} {},{} A{},{} 0 0,{} {},{} A{},{} 0 0,{} {},{}" fill="none" stroke="#3030d0" stroke-width="0.001" />'.format(p1[0],p1[1],
				r1,r1,b1,p2[0],p2[1],
				r2,r2,b2,p3[0],p3[1],
				r3,r3,b3,p1[0],p1[1])

#def computeBisector(p1,p2):
#	x1 = p1[0];
#	x2 = p2[0];
#	y1 = p1[1];
#	y2 = p2[1];
#	r02= (1 - x2**2 - y2**2)/( 1 - x1**2 - y1**2);
#	
#	xc = -(r02 * x1 - x2)/( 1 - r02);
#	yc = -(r02 * y1 - y2)/( 1 - r02);

#	#r2 = ( 2*x2*r02*x1 - r02*(x1**2 + x2**2) )/ (( 1 - r02) ** 2)
#	#	+( 2*y2*r02*y1 - r02*(y1**2 + y2**2) )/ (( 1 - r02) ** 2);
#	r2 = r02*( (x1 - x2)**2 + (y1 - y2)**2 )/ (( 1 - r02) ** 2);

def computeBisector(p1,p2):
	x1 = p1[0];
	x2 = p2[0];
	y1 = p1[1];
	y2 = p2[1];
	n1 = x1**2 + y1 ** 2;
	n2 = x2**2 + y2 ** 2;
	
	xc = ( (1-n1)*x2 - (1-n2)*x1)/( n2 - n1 );
	yc = ( (1-n1)*y2 - (1-n2)*y1)/( n2 - n1 );

	#r2 = ( 2*x2*r02*x1 - r02*(x1**2 + x2**2) )/ (( 1 - r02) ** 2)
	#	+( 2*y2*r02*y1 - r02*(y1**2 + y2**2) )/ (( 1 - r02) ** 2);
	r2 = ( (x1 - x2)**2 + (y1 - y2)**2 )*( 1 - n1)*( 1 - n2) / ( (n2 - n1) ** 2);
	return( np.array( (xc,yc,np.sqrt(abs(r2)))) );

#finds the point at which two circles intersect and returns the one in the disk
def getInteriorVertex(c1,c2):
	z1 = c1[0] + c1[1]*1.0j;
	z2 = c2[0] + c2[1]*1.0j;
	r1 = c1[2];
	r2 = c2[2];

        #circles do not intersect
	if (r1 + r2) < ( abs(z1 - z2) ):
		return 0;
	if abs(r2-r1) >  ( abs(z1 - z2) ):
		return 0;
        #circles are the same
        if abs(z1 - z2) == 0:
            return 0;

	t = (abs(z1 - z2)**2 + r1**2 - r2**2)/2/abs(z1-z2);

        #sanity check?
	#if abs(r1) < abs(t):
	#	return 0;

	w1 = z1 + (z2-z1)/(abs(z1 - z2))*( t + 1.0j*np.sqrt(abs(abs(r1) ** 2 - abs(t) ** 2)));
	w2 = z1 + (z2-z1)/(abs(z1 - z2))*( t - 1.0j*np.sqrt(abs(abs(r1) ** 2 - abs(t) ** 2)));

	return( (w1.real,w1.imag) if abs(w1) < abs(w2) else (w2.real,w2.imag) );

def getPlusBoundaryVertex(c1):
	z1 = c1[0] + c1[1]*1.0j;
	z2 = 0;
	r1 = c1[2];
	r2 = 1;

	t = (abs(z1 - z2)**2 + r1**2 - r2**2)/2/abs(z1-z2);

	w1 = z1 + (z2-z1)/(abs(z1 - z2))*( t + 1.0j*np.sqrt(abs(r1) ** 2 - abs(t) ** 2));

	return( (w1.real,w1.imag));

def getMinusBoundaryVertex(c1):
	z1 = c1[0] + c1[1]*1.0j;
	z2 = 0;
	r1 = c1[2];
	r2 = 1;

	t = (abs(z1 - z2)**2 + r1**2 - r2**2)/2/abs(z1-z2);

	w1 = z1 + (z2-z1)/(abs(z1 - z2))*( t - 1.0j*np.sqrt(abs(r1) ** 2 - abs(t) ** 2));

	return( (w1.real,w1.imag));

def circleContains(c, x):
        return( (c[2] ** 2) > ( (x[0]-c[0])**2 + (x[1]-c[1])**2));

#check if the disks c1 and c2 do not overlap
def circlesDisjoint(c1,c2):
	z1 = c1[0] + c1[1]*1.0j;
	z2 = c2[0] + c2[1]*1.0j;
	r1 = c1[2];
	r2 = c2[2];

        #circles are too far apart
	if (r1 + r2) < ( abs(z1 - z2) ):
		return True;
        return False;

#check if c2 fits inside c1
def circleNests(c1,c2):
	z1 = c1[0] + c1[1]*1.0j;
	z2 = c2[0] + c2[1]*1.0j;
	r1 = c1[2];
	r2 = c2[2];

        #the radius of c1 is so large it contains c2
	if (r1 > (r2 + ( abs(z1 - z2) ))):
		return True;
        return False;


	
def drawCell(p, boundary):
	d = boundary.shape[0]
	unshadowed = np.ones(d,dtype=bool);
	circs = np.zeros((d,3));

        #compute hyperbolic halfplanes that (could) bound the cell
	for i in range(0,d):
		circs[i] = computeBisector(p,boundary[i]);
#                print '<circle cx="{}" cy="{}" r="{}" fill="none" stroke="purple" stroke-width="0.001" />'.format(circs[i,0],circs[i,1],circs[i,2]);

        #remove bisectors for which there is another bisector separating it from the nucleus
        #also remove bisectors that are disjoint from a halfspace containing the nucleus
	for i in range(0,d):
            unshadowed[i] = True;
	    for j in range(0,d):
                dist = abs(circs[j,0] - circs[i,0] + 1.0j*(circs[j,1]-circs[i,1]))
#                if( circs[i,2] > (circs[j,2] + dist) and i != j):
#                    unshadowed[i] = False;
                if( circlesDisjoint(circs[i],circs[j]) == True and i != j):
                    if( circleContains(circs[j],p)):
                        unshadowed[i] = False;
                if( circleNests(circs[i],circs[j]) == True and i != j):
                    if( circleContains(circs[j],p)):
                        unshadowed[i] = False;

        boundary = boundary[unshadowed]
        circs = circs[unshadowed]
	d = circs.shape[0]

	arg = np.zeros(d);
	orientation = np.zeros(d);
	verts = np.zeros((d,3));

        #order the circles according to a clockwise ordering
	for i in range(0,d):
                z0 = p[0]+p[1]*1.0j;
                z = boundary[i,0] + boundary[i,1]*1.0j;
                u = (z-z0)/(1-complex.conjugate(z0)*z);
		arg[i] = -math.atan2(u.imag, u.real); 
	inds = arg.argsort();
	circs = circs[inds];

	for i in range(0,d):
		x = getInteriorVertex(circs[i], circs[(i+1) % d]);
#                print '<circle cx="{}" cy="{}" r="{}" fill="none" stroke="purple" stroke-width="{}" />'.format(circs[i,0],circs[i,1],circs[i,2],float(i+1)*0.002);
		if x == 0:
                        #the circles did not cross
		        verts[i] = [0,0,1];
                else:
                        verts[i] = [x[0],x[1],0];
#                        print '<circle cx="{}" cy="{}" r="0.01" fill="none" stroke="green" stroke-width="0.05" />'.format(x[0],x[1])
                        #print '<text cx="{}" cy="{}" fill="green" stroke-width="0.05">{}</text>'.format(x[0],x[1],'1')
#                        print '<text x="{}" y="{}" font-size="0.1" text-anchor="middle" fill="green">{}</text>'.format(x[0],x[1],i)
                        #the intersection may be exterior, so check
                        runs = np.array(range(0,d));
                        runs = runs[runs != i]
                        runs = runs[runs != ( (i+1)%d)]
                        for j in runs:
                            dist = abs(circs[j,0] - x[0] + 1.0j*(circs[j,1]-x[1]))
                            if( dist > circs[j,2] and circleContains(circs[j],p)):
		                verts[i] = [0,0,1];
#                                print '<circle cx="{}" cy="{}" r="0.001" fill="none" stroke="green" stroke-width="0.05" />'.format(x[0],x[1])
                        
                        #it can happen if d=2 that we get the same point twice
                        if( d == 2 and i == 1):
                            verts[1] = [0,0,1];


	for i in range(0,d):
                #we add arcs that run to the boundary from an interior point
                if (verts[i,2] == 0 and verts[ (i+1)%d,2] == 1):
                        #compute orientation of the crossing between the two circles
                        b1 = getPlusBoundaryVertex(circs[i])
                        b2 = getPlusBoundaryVertex(circs[(i+1)%d])
                        x1 = b1[0] - verts[i][0];
                        x2 = b2[0] - verts[i][0];
                        y1 = b1[1] - verts[i][1];
                        y2 = b2[1] - verts[i][1];
                        #according to the orientation of the crossing
                        #and the location of the point with respect to the first circle
                        #the arc
                        #of the second circle adjacent to the point is either the positive
                        #or negative
                        if (x1*y2 - x2*y1) > 0:
                            if (((p[0] - circs[i,0]) ** 2 + (p[1] - circs[i,1]) ** 2) > (circs[i,2]**2)):
                                x = getMinusBoundaryVertex(circs[(i+1)%d])
                            else:
                                x = getPlusBoundaryVertex(circs[(i+1)%d])
                            verts[(i+1) % d,0] = x[0]
                            verts[(i+1) % d,1] = x[1]
                        else:
                            if (((p[0] - circs[i,0]) ** 2 + (p[1] - circs[i,1]) ** 2) > (circs[i,2]**2)):
                                x = getPlusBoundaryVertex(circs[(i+1)%d])
                            else:
                                x = getMinusBoundaryVertex(circs[(i+1)%d])
                            verts[(i+1) % d,0] = x[0]
                            verts[(i+1) % d,1] = x[1] 
#                        print '<circle cx="{}" cy="{}" r="0.01" fill="none" stroke="green" stroke-width="0.05" />'.format(x[0],x[1])
#

	for i in range(0,d):
                #if (verts[i,2] == 0 or verts[ (i+1)%d,2] == 0):
       		f = np.linalg.det( [[ verts[i,0], verts[i,1], 1],
	        		[ verts[(i+1)%d,0], verts[(i+1)%d,1], 1],
		        	[ circs[(i+1)%d,0], circs[(i+1)%d,1], 1]]);
       		orientation[i] = 1 if f > 0 else 0;

	for i in range(0,d):
                if (verts[i,2] == 0):
	                print '<path d="M{},{}'.format(verts[i][0],verts[i][1]);
		        print 'A{},{} 0 0,{:d} {},{}'.format(circs[(i+1) % d,2], circs[(i+1)%d,2], int(orientation[i]), verts[(i+1) %d,0], verts[(i+1) % d,1]);
                        print '" fill="none" stroke="#f03030" stroke-width="0.005" />'

                #we add arcs that run from the boundary to an interior point
                if (verts[i,2] == 1 and verts[ (i+1)%d,2] == 0):
                        #compute orientation of the crossing between the two circles
                        b1 = getPlusBoundaryVertex(circs[i])
                        b2 = getPlusBoundaryVertex(circs[(i+1)%d])
                        x1 = b1[0] - verts[i][0];
                        x2 = b2[0] - verts[i][0];
                        y1 = b1[1] - verts[i][1];
                        y2 = b2[1] - verts[i][1];
                        #according to the orientation of the crossing
                        #and the location of the point with respect to the first circle
                        #the arc
                        #of the second circle adjacent to the point is either the positive
                        #or negative
                        if circleContains(circs[(i+1)%d],p):
                            if circleContains(circs[i],p):
                                x = getPlusBoundaryVertex(circs[(i+1)%d])
                            else:
                                x = getPlusBoundaryVertex(circs[(i+1)%d])
                        else:
                            if circleContains(circs[i],p):
                                x = getMinusBoundaryVertex(circs[(i+1)%d])
                            else:
                                x = getMinusBoundaryVertex(circs[(i+1)%d])

                        f = np.linalg.det( [[ x[0], x[1], 1],
                                [verts[(i+1) %d,0], verts[(i+1) % d,1],1],
		        	[ circs[(i+1)%d,0], circs[(i+1)%d,1], 1]]);
                        orientation[i] = 1 if f > 0 else 0;
                        print '<path d="M{},{}'.format(x[0],x[1]);
		        print 'A{},{} 0 0,{:d} {},{}'.format(circs[(i+1) % d,2], circs[(i+1)%d,2], int(orientation[i]),verts[(i+1) %d,0], verts[(i+1) % d,1]);
                        print '" fill="none" stroke="#f03030" stroke-width="0.005" />'

                #we add arcs that run to the boundary from boundary
                if (verts[i,2] == 1 and verts[ (i+1)%d,2] == 1):
                    if( (circs[(i+1) % d,2] ** 2) > ( (p[0]-circs[(i+1) % d,0])**2 + (p[1]-circs[(i+1) % d,1])**2)):
                        x0 = getMinusBoundaryVertex(circs[(i+1)%d])
                        x1 = getPlusBoundaryVertex(circs[(i+1)%d])
                        #verts[i,0] = x0[0]
                        #verts[i,1] = x0[1]
                        #verts[(i+1) % d,0] = x1[0]
                        #verts[(i+1) % d,1] = x1[1]
                        f = np.linalg.det( [[ x0[0], x0[1], 1],
	        		[ x1[0], x1[1], 1],
		        	[ circs[(i+1)%d,0], circs[(i+1)%d,1], 1]]);
                        orientation[i] = 1 if f > 0 else 0;
                        print '<path d="M{},{}'.format(x0[0],x0[1]);
		        print 'A{},{} 0 0,{:d} {},{}'.format(circs[(i+1) % d,2], circs[(i+1)%d,2], int(orientation[i]), x1[0], x1[1]);
                        print '" fill="none" stroke="#f03030" stroke-width="0.005" />'

                	
        print '<circle cx="0" cy="0" r="1" fill="none" stroke="green" stroke-width="0.001" />'
def initSVG():
	print '<?xml version="1.0" standalone="no"?>'
	print '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
	print '<svg width="10cm" height="10cm" viewBox="-1 -1 2 2" xmlns="http://www.w3.org/2000/svg" version="1.1">'
	return;

def closeSVG():
	print '</svg>'


#points = PPP(0.2,0.9995)
#points = PPP(1.0,0.9975)

#n=10
#np.random.seed(1234)

n=1000
np.random.seed(123456)

coeffs=(np.random.randn(n) + np.random.randn(n)*1.0j)/np.sqrt(2);
#coeffs[n-1]=0.001
cpoints = np.roots(coeffs)
cpoints = cpoints[ abs(cpoints) < 1.00]

points = np.column_stack((cpoints.real,cpoints.imag))
tri = Delaunay(points);

initSVG();

#print '<circle cx="0" cy="0" r="1" stroke="blue" stroke-width="0.01" fill="none" />'


ind, indp = tri.vertex_neighbor_vertices;
#for i in range(0,10):#range(0,len(points)):
for i in range(0,len(points)):#range(0,len(points)):
#    print '<circle cx="{}" cy="{}" r="0.01" fill="none" stroke="green" stroke-width="0.01" />'.format(points[i,0],points[i,1])
    drawCell( points[i], points[indp[ind[i]:ind[i+1]]])

for i in range(0,len(tri.simplices)):
	if checkFinite(points[tri.simplices[i]]):
		drawTri(points[tri.simplices[i]]);

#print '<text x="{}" y="{}" font-size="0.1" text-anchor="middle" fill="green">{}</text>'.format(0,0,len(cpoints))
closeSVG();



