#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "blas.h"
#include "gp.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include <Windows.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define FRAMESl 3
#define FRAMESr 3
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
#include "http_stream.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probsl, **probsr;
static box *boxesl, *boxesr;
static network netl, netr;
static image in_sl, in_sr;
static image det_sl, det_sr;
static CvCapture *capl, *capr;
static int cpp_video_capture = 0;

static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;

static float *predictionsl[FRAMESl], *predictionsr[FRAMESr];
static int demo_indexl = 0;
static int demo_indexr = 0;
static image imagesl[FRAMESl], imagesr[FRAMESr];
static IplImage* ipl_imagesl[FRAMESl], *ipl_imagesr[FRAMESr];
static float *avgl, *avgr;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void draw_detections_cv_v3(IplImage* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output);
cube_center draw_detections_cv_v4(IplImage* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close, const CvArr *mapx, const CvArr *mapy);
image get_image_from_stream_letterbox(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close, const CvArr *mapx, const CvArr *mapy);
int get_stream_fps(CvCapture *cap, int cpp_video_capture);
IplImage* in_imgl, *in_imgr;
IplImage* det_imgl, *det_imgr;
IplImage* show_imgl, *show_imgr;
int clr;
static int flag_exit;
static int letter_box = 0;
static CvScalar  color[12] = {
	{ 0,132,8,0 },//red1
	{ 30,255,255,0 },
	{ 111,143,0,0 },  //blue2
	{ 134,255,255,0 },
	{ 53,103,14,0 },  //green3
	{ 95,149,130,0 },
	{ 12,220,112,0 }, //yellow4
	{ 26,226,255,0 },
	{ 0,14,0,0 },     //black5
	{ 106,27,72,0 },
	{ 137,43,255,0 }, //pink6
	{ 241,163,255,0 } };

//void hough(int clr)
void hough(IplImage*orl, IplImage*orr, int clr)
{
	free_image(det_sl); free_image(det_sr);
	

	clr = clr - 1;
	clr = clr * 2;
	int flag = 0;
	CvPoint cen;
	int max_pix = 0;
//	IplImage *orl = cvCreateImage(cvGetSize(det_imgl), 8, 3), *orr=cvCreateImage(cvGetSize(det_imgr), 8, 3);
//	cvCopy(det_imgl, orl,0); cvCopy(det_imgr, orr, 0);
	IplImage *andl = cvCreateImage(cvGetSize(orl), 8, 3),
		*srccopyl = cvCreateImage(cvGetSize(orl), 8, 3),
		*andr = cvCreateImage(cvGetSize(orr), 8, 3),
		*srccopyr = cvCreateImage(cvGetSize(orr), 8, 3);
	IplImage *roil = cvCreateImage(cvGetSize(orl), 8, 1),
		*smol = cvCreateImage(cvGetSize(orl), 8, 1),
		*pixl = cvCreateImage(cvGetSize(orl), 8, 1),
		*roir = cvCreateImage(cvGetSize(orr), 8, 1),
		*smor = cvCreateImage(cvGetSize(orr), 8, 1),
		*pixr = cvCreateImage(cvGetSize(orr), 8, 1);
//	srccopyl = orl; srccopyr = orr;
	cvCopy(orl, srccopyl, 0);
	cvCopy(orr, srccopyr, 0);
	cvCvtColor(orl, smol, CV_RGB2GRAY); cvCvtColor(orr, smor, CV_RGB2GRAY);
	CvMemStorage* storagel = cvCreateMemStorage(0);
	CvMemStorage* storager = cvCreateMemStorage(0);
	CvSeq* resultsl = cvHoughCircles(smol, storagel, CV_HOUGH_GRADIENT, 2, 20, 100, 90, 30, 45);
	CvSeq* resultsr = cvHoughCircles(smor, storager, CV_HOUGH_GRADIENT, 2, 20, 100, 90, 30, 45);
	
	for (int i = 0; i < resultsl->total; i++) {
		cvSetZero(andl);
		cvSetZero(roil);
		cvSetZero(pixl); 
		float*pl = (float*)cvGetSeqElem(resultsl, i);
		CvPoint ptl = cvPoint(cvRound(pl[0]), cvRound(pl[1]));
		cvCircle(orl, ptl, cvRound(pl[2]), CV_RGB(0xff, 0x00, 0x00), 3, 8, 0);
		cvCircle(roil, ptl, cvRound(pl[2]), CV_RGB(0xff, 0xff, 0xff), -1, 8, 0);
		cvAnd(srccopyl, srccopyl, andl, roil);
		cvCvtColor(andl, andl, CV_BGR2HSV);
		cvInRangeS(andl, color[clr], color[clr + 1], pixl);
		int count_pixl = cvCountNonZero(pixl);
		if (max_pix < count_pixl) {
			max_pix = count_pixl;
			cen = ptl;
			flag = 1;
		}
	}
	for (int i = 0; i < resultsr->total; i++) {
		cvSetZero(andr);
		cvSetZero(roir);
		cvSetZero(pixr);
		float*pr = (float*)cvGetSeqElem(resultsr, i); 
		CvPoint ptr = cvPoint(cvRound(pr[0]), cvRound(pr[1]));
		cvCircle(orr, ptr, cvRound(pr[2]), CV_RGB(0xff, 0x00, 0x00), 3, 8, 0);
		cvCircle(roir, ptr, cvRound(pr[2]), CV_RGB(0xff, 0xff, 0xff), -1, 8, 0);
		cvAnd(srccopyr, srccopyr, andr, roir);
		cvCvtColor(andl, andl, CV_BGR2HSV);
		cvInRangeS(andl, color[clr], color[clr + 1], pixl); 
		int count_pixr = cvCountNonZero(pixl); 
		if (max_pix < count_pixr) {
			max_pix = count_pixr;
			cen = ptr;
			flag = 2;
		}
	}

	switch (flag)
	{
	case 0:
		break;
	case 1:cvCircle(orl, cen, 10, CV_RGB(0xff, 0x00, 0x00), -1, 8, 0); printf("l : x=%d,y=%d\n", cen.x, cen.y);
		WriteC(1, cen.x, cen.y);
		break;
	case 2:cvCircle(orr, cen, 10, CV_RGB(0xff, 0x00, 0x00), -1, 8, 0); printf("r : x=%d,y=%d\n", cen.x, cen.y);
		WriteC(2, cen.x, cen.y);
		break;
	}
	cvShowImage("Demol", orl);
	cvShowImage("Demor", orr);
	cvReleaseImage(&orl); cvReleaseImage(&orr); 
	cvReleaseImage(&andl); cvReleaseImage(&srccopyl);
	cvReleaseImage(&roil); cvReleaseImage(&smol); cvReleaseImage(&pixl);
	cvReleaseImage(&andr); cvReleaseImage(&srccopyr);
	cvReleaseImage(&roir);  cvReleaseImage(&smor); cvReleaseImage(&pixr);
	cvReleaseMemStorage(&storagel); cvReleaseMemStorage(&storager);
	cvReleaseImage(&orl); cvReleaseImage(&orr);
}

void *fetch_in_threadl(void *ptr, const CvArr *mapx, const CvArr *mapy)
{
	int dont_close_stream = 1;    // set 1 if your IP-camera periodically turns off and turns on video-stream
	if (letter_box) {
		in_sl = get_image_from_stream_letterbox(capl, netl.w, netl.h, netl.c, &in_imgl, cpp_video_capture, dont_close_stream, mapx, mapy);
	}
	else {
		in_sl = get_image_from_stream_resize(capl, netl.w, netl.h, netl.c, &in_imgl, cpp_video_capture, dont_close_stream, mapx, mapy);
	}
	if (!in_sl.data) {
		printf("Stream caml closed.\n");
		flag_exit = 1;
		return EXIT_FAILURE;
	}
	return 0;
}
void *fetch_in_threadr(void *ptr, const CvArr *mapx, const CvArr *mapy)
{
	
	int dont_close_stream = 1;
	if (letter_box) {
		in_sr = get_image_from_stream_letterbox(capr, netr.w, netr.h, netr.c, &in_imgr, cpp_video_capture, dont_close_stream, mapx, mapy);
	}
	else {

		in_sr = get_image_from_stream_resize(capr, netr.w, netr.h, netr.c, &in_imgr, cpp_video_capture, dont_close_stream, mapx, mapy);
	}
	if (!in_sr.data) {
		printf("Stream camr closed.\n");
		flag_exit = 1;
		return EXIT_FAILURE;
	}


	return 0;
}
void *detect_in_threadl(void *ptr)
{
	cube_center lcenter;
	float nms = .45;    // 0.4F
	layer l = netl.layers[netl.n - 1];
	float *Xl = det_sl.data;
	float *predictionl = network_predict(netl, Xl);
	memcpy(predictionsl[demo_indexl], predictionl, l.outputs * sizeof(float));
	mean_arrays(predictionsl, FRAMESl, l.outputs, avgl);
	l.output = avgl;
	free_image(det_sl);

	int nboxesl = 0;
	detection *detsl = NULL;
	if (letter_box) {
		detsl = get_network_boxes(&netl, in_imgl->width, in_imgl->height, demo_thresh, demo_thresh, 0, 1, &nboxesl, 1); // letter box
	}
	else {
		detsl = get_network_boxes(&netl, det_sl.w, det_sl.h, demo_thresh, demo_thresh, 0, 1, &nboxesl, 0); // resized
	}
	if (nms) {
		do_nms_sort(detsl, nboxesl, l.classes, nms);
	}
	ipl_imagesl[demo_indexl] = det_imgl;
	det_imgl = ipl_imagesl[(demo_indexl + FRAMESl / 2 + 1) % FRAMESl];
	demo_indexl = (demo_indexl + 1) % FRAMESl;
	lcenter=draw_detections_cv_v4(det_imgl, detsl, nboxesl, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
	if (0!=nboxesl){
	printf("lx=%d ly=%d nb=%d\n", lcenter.x, lcenter.y,nboxesl);
	WriteC(1, lcenter.x, lcenter.y);
	free_detections(detsl, nboxesl);
	}
	return 0;
}
void *detect_in_threadr(void *ptr)
{
	cube_center rcenter;
	float nms = .45;    // 0.4F

	layer r = netr.layers[netr.n - 1];
	float *Xr = det_sr.data;
	float *predictionr = network_predict(netr, Xr);
	memcpy(predictionsr[demo_indexr], predictionr, r.outputs * sizeof(float));
	mean_arrays(predictionsr, FRAMESr, r.outputs, avgr);
	r.output = avgr;
	free_image(det_sr);

	int nboxesr = 0;
	detection *detsr = NULL;
	if (letter_box) {
		detsr = get_network_boxes(&netl, in_imgr->width, in_imgr->height, demo_thresh, demo_thresh, 0, 1, &nboxesr, 1);
	}
	else {
		detsr = get_network_boxes(&netr, det_sr.w, det_sr.h, demo_thresh, demo_thresh, 0, 1, &nboxesr, 0);
	}
	if (nms) {
		do_nms_sort(detsr, nboxesr, r.classes, nms);
	}

	ipl_imagesr[demo_indexr] = det_imgr;
	det_imgr = ipl_imagesr[(demo_indexr + FRAMESr / 2 + 1) % FRAMESr];
	demo_indexr = (demo_indexr + 1) % FRAMESr;

	rcenter=draw_detections_cv_v4(det_imgr, detsr, nboxesr, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
	if (0 != nboxesr) {
		printf("rx=%d ry=%d nb=%d\n", rcenter.x, rcenter.y, nboxesr);
		WriteC(2, rcenter.x, rcenter.y);
	}
	free_detections(detsr, nboxesr);

	return 0;
}
double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_indexl, int cam_indexr, const char *filename, char **names, int classes,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
	
	InitC();
	
	image **alphabet = load_alphabet();
	cvNamedWindow("bar", CV_WINDOW_NORMAL);
	int positions = 0;
	cvCreateTrackbar("color", "bar", &positions, 7, NULL);
	int delay = frame_skip;
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_ext_output = ext_output;

	CvMat *intrinsic = cvCreateMat(3, 3, CV_64FC1);
	CvMat *distortion = cvCreateMat(5, 1, CV_64FC1);
	cvZero(intrinsic);
	cvZero(distortion);
	cvmSet(intrinsic, 0, 0, 552.0429858016315);
	cvmSet(intrinsic, 0, 1, 0);
	cvmSet(intrinsic, 0, 2, 350.4666686405849);
	cvmSet(intrinsic, 1, 0, 0);
	cvmSet(intrinsic, 1, 1, 550.2633757205168);
	cvmSet(intrinsic, 1, 2, 281.6993627734493);
	cvmSet(intrinsic, 2, 0, 0);
	cvmSet(intrinsic, 2, 1, 0);
	cvmSet(intrinsic, 2, 2, 1);
	cvmSet(distortion, 0, 0, -0.4545097535530283);
	cvmSet(distortion, 1, 0, 0.2993757533948309);
	cvmSet(distortion, 2, 0, 0.0008457326274740741);
	cvmSet(distortion, 3, 0, 0.001468986057480372);
	cvmSet(distortion, 4, 0, -0.1384713052343635);
	const CvArr *mapx = cvCreateImage(cvSize(640, 480), IPL_DEPTH_32F, 1);
	const CvArr *mapy = cvCreateImage(cvSize(640, 480), IPL_DEPTH_32F, 1);
	cvInitUndistortMap(intrinsic, distortion, mapx, mapy);
	netl = parse_network_cfg_custom(cfgfile, 1);    // set batch=1
	netr = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile) {
		load_weights(&netl, weightfile);
		load_weights(&netr, weightfile);
	}

	fuse_conv_batchnorm(netl);
	fuse_conv_batchnorm(netr);
	srand(2222222);

	if (filename) {
		printf("video file: %s\n", filename);
		cpp_video_capture = 1;
		capl = get_capture_video_stream(filename);
		capr = get_capture_video_stream(filename);
	}
	else {
		printf("Webcam indexl: %d\n", cam_indexl);
		printf("Webcam indexr: %d\n", cam_indexr);
		cpp_video_capture = 1;
		capl = get_capture_webcam(cam_indexl);
		capr = get_capture_webcam(cam_indexr);
	}
	if (!capl) {
#ifdef WIN32
		printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
		error("Couldn't connect to left cam.\n");
	}
	if (!capr)
	{
#ifdef WIN32
		printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
		error("Couldn't connect to right cam.\n");
	}
	layer l = netl.layers[netl.n - 1];
	layer r = netl.layers[netr.n - 1];
	int j;
	avgl = (float *)calloc(l.outputs, sizeof(float));
	avgr = (float *)calloc(r.outputs, sizeof(float));
	for (j = 0; j < FRAMESl; ++j) predictionsl[j] = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < FRAMESr; ++j) predictionsr[j] = (float *)calloc(r.outputs, sizeof(float));
	for (j = 0; j < FRAMESl; ++j) imagesl[j] = make_image(1, 1, 3);
	for (j = 0; j < FRAMESr; ++j) imagesr[j] = make_image(1, 1, 3);
	boxesl = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	boxesr = (box *)calloc(r.w*r.h*r.n, sizeof(box));
	probsl = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	probsr = (float **)calloc(r.w*r.h*r.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probsl[j] = (float *)calloc(l.classes, sizeof(float *));
	for (j = 0; j < r.w*r.h*r.n; ++j) probsr[j] = (float *)calloc(l.classes, sizeof(float *));

	flag_exit = 0;

	pthread_t fetch_threadl, fetch_threadr;
	pthread_t detect_threadl, detect_threadr;
	fetch_in_threadl(0, mapx, mapy);
	det_imgl = in_imgl;
	fetch_in_threadr(0, mapx, mapy);
	det_imgr = in_imgr;
	show_imgl = cvCreateImage(cvGetSize(det_imgl), 8, 3);
	show_imgr = cvCreateImage(cvGetSize(det_imgr), 8, 3);
	det_sl = in_sl;
	det_sr = in_sr;
	fetch_in_threadl(0, mapx, mapy); fetch_in_threadr(0, mapx, mapy);
	detect_in_threadl(0);
	detect_in_threadr(0);
	det_imgl = in_imgl;
	det_imgr = in_imgr;
	det_sl = in_sl;
	det_sr = in_sr;
	for (j = 0; j < FRAMESl / 2; ++j) {
		fetch_in_threadl(0, mapx, mapy);
		detect_in_threadl(0);
		det_imgl = in_imgl;
		det_sl = in_sl;
	}
	for (j = 0; j < FRAMESr / 2; ++j) {
		fetch_in_threadr(0, mapx, mapy);
		detect_in_threadr(0);
		det_imgr = in_imgr;
		det_sr = in_sr;
	}
	int count = 0;
	if (!prefix && !dont_show) {
		cvNamedWindow("Demol", CV_WINDOW_NORMAL);
		cvNamedWindow("Demor", CV_WINDOW_NORMAL);
		cvMoveWindow("Demol", 0, 0);
		cvMoveWindow("Demor", 1000, 0);
		cvResizeWindow("Demol", 640, 480);
		cvResizeWindow("Demor", 640, 480);
	}

	CvVideoWriter* output_video_writer = NULL;    // cv::VideoWriter output_video;
	if (out_filename && !flag_exit)
	{
		CvSize sizel, sizer;
		sizel.width = det_imgl->width, sizel.height = det_imgl->height;
		sizer.width = det_imgr->width, sizer.height = det_imgr->height;
		int src_fps_left = 25;
		int src_fps_right = 25;
		src_fps_left = get_stream_fps(capl, cpp_video_capture);
		src_fps_right = get_stream_fps(capr, cpp_video_capture);
		//const char* output_name = "test_dnn_out.avi";
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('H', '2', '6', '4'), src_fps, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), src_fps_left, sizel, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'J', 'P', 'G'), src_fps, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', 'V'), src_fps, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', '2'), src_fps, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('X', 'V', 'I', 'D'), src_fps, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('W', 'M', 'V', '2'), src_fps, size, 1);
	}

	double before = get_wall_time();
	while (1) {
		++count;
		if (0) {//in1 ..
			printf("start                  while                     yeah");
			if (pthread_create(&fetch_threadl, 0, fetch_in_threadl, 0)) error("Threadl creation failed");//∂¡»°ÕºœÒ
			if (pthread_create(&detect_threadl, 0, detect_in_threadl, 0)) error("Threadl creation failed");//ºÏ≤‚ŒÔÃÂ
			if (pthread_create(&fetch_threadr, 0, fetch_in_threadr, 0)) error("Threadr creation failed");
			if (pthread_create(&detect_threadr, 0, detect_in_threadr, 0)) error("Threadr creation failed");

			if (!dont_show) {//in1 ..
				show_image_cv_ipl(show_imgl, "Demol");
				show_image_cv_ipl(show_imgr, "Demor");
				int c = cvWaitKey(1);
				if (c == 10) {//nop1  ..
					if (frame_skip == 0) frame_skip = 60;
					else if (frame_skip == 4) frame_skip = 0;
					else if (frame_skip == 60) frame_skip = 4;
					else frame_skip = 0;
				}
				else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
				{//nop1 ..
					flag_exit = 1;
				}
			}
			cvReleaseImage(&show_imgl);
			cvReleaseImage(&show_imgr);
			pthread_join(fetch_threadl, 0);
			pthread_join(detect_threadl, 0);
			pthread_join(fetch_threadr, 0);
			pthread_join(detect_threadr, 0);
			if (flag_exit == 1) break;

			if (delay == 0) {//in1 ..
				show_imgl = det_imgl;
				show_imgr = det_imgr;
			}
			det_imgl = in_imgl;
			det_imgr = in_imgr;
			det_sl = in_sl;
			det_sr = in_sr;
		}
		else {//nop1
			  //			printf("dont know wheter in else or not and give a remarkable sign like alot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			fetch_in_threadl(0, mapx, mapy); fetch_in_threadr(0, mapx, mapy);
			det_imgl = in_imgl;
			det_imgr = in_imgr;
			clr = cvGetTrackbarPos("color", "bar");
			if (clr == 0) {
				det_sl = in_sl;
				det_sr = in_sr;
				detect_in_threadl(0);
				detect_in_threadr(0);
				show_imgl = det_imgl;
				show_imgr = det_imgr;
				show_image_cv_ipl(show_imgl, "Demol");
				show_image_cv_ipl(show_imgr, "Demor");
			}//else {hough(det_imgl, det_imgr, clr);
//				int l;
			else {
				det_sl = in_sl;
				det_sr = in_sr;
				hough(det_imgl, det_imgr, clr);
//				cvReleaseImage(&det_imgl);
//				cvReleaseImage(&det_imgl);

				
			}
			if (1) {//nop1


				cvWaitKey(1);
			}
			cvReleaseImage(&show_imgl);
			cvReleaseImage(&show_imgr);
		}
		--delay;
		if (delay < 0) {//in1 ..
			delay = frame_skip;

			double after = get_wall_time();
			float curr = 1. / (after - before);
			fps = curr;
			before = after;
		}
//		printf("x=%d  y=%d", cx, cy);
	}
	printf("input video stream closed. \n");
	if (output_video_writer) {//nop1
		cvReleaseVideoWriter(&output_video_writer);
		printf("output_video_writer closed. \n");
	}
	// free memory
	cvReleaseImage(&show_imgl); cvReleaseImage(&show_imgr);
	cvReleaseImage(&in_imgl); cvReleaseImage(&in_imgr);
	free_image(in_sl); free_image(in_sr);
	free(avgl);
	for (j = 0; j < FRAMESl; ++j) free(predictionsl[j]);
	for (j = 0; j < FRAMESl; ++j) free_image(imagesl[j]);
	for (j = 0; j < l.w*l.h*l.n; ++j) free(probsl[j]);
	free(boxesl);
	free(probsl);

	free_ptrs(names, netl.layers[netl.n - 1].classes);

	free(avgr);
	for (j = 0; j < FRAMESr; ++j) free(predictionsr[j]);
	for (j = 0; j < FRAMESr; ++j) free_image(imagesr[j]);

	for (j = 0; j < r.w*r.h*r.n; ++j) free(probsr[j]);
	free(boxesr);
	free(probsr);

	free_ptrs(names, netr.layers[netr.n - 1].classes);

	int i;
	const int nsize = 8;
	for (j = 0; j < nsize; ++j) {
		for (i = 32; i < 127; ++i) {
			free_image(alphabet[j][i]);
		}
		free(alphabet[j]);
	}
	free(alphabet);

	free_network(netl);
	free_network(netr);
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_indexl, int cam_indexr, const char *filename, char **names, int classes,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

