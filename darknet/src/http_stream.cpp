#ifdef OPENCV
//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

//
// socket related abstractions:
//
#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#include <winsock.h>
#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <time.h>
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
    WSADATA w;
    _INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;
#else       /* ! win32 */
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
#endif /* _WIN32 */

#include <cstdio>
#include <vector>
#include <iostream>
using std::cerr;
using std::endl;
unsigned char d[8];
int len;
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#endif
using namespace cv;

#include "http_stream.h"
#include "image.h"


class MJPGWriter
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout millis.
    int quality; // jpeg compression [1..100]

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    MJPGWriter(int port = 0, int _timeout = 200000, int _quality = 30)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
        , quality(_quality)
    {
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~MJPGWriter()
    {
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error : couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error : couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(const Mat & frame)
    {
        fd_set rread = master;
        struct timeval to = { 0,timeout };
        if (::select(maxfd+1, &rread, NULL, NULL, &to) <= 0)
            return true; // nothing broken, there's just noone listening

        std::vector<uchar> outbuf;
        std::vector<int> params;
        params.push_back(IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        cv::imencode(".jpg", frame, outbuf, params);
        size_t outlen = outbuf.size();

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s<=maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error : couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\r\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\r\n"
                    "Accept-Range: bytes\r\n"
                    "Connection: close\r\n"
                    "Max-Age: 0\r\n"
                    "Expires: 0\r\n"
                    "Cache-Control: no-cache, private\r\n"
                    "Pragma: no-cache\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
                    "\r\n", 0);
                cerr << "new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                char head[400];
                sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen);
                _write(s, head, 0);
                int n = _write(s, (char*)(&outbuf[0]), outlen);
                //cerr << "known client " << s << " " << n << endl;
                if (n < outlen)
                {
                    cerr << "kill client " << s << endl;
                    ::shutdown(s, 2);
                    FD_CLR(s, &master);
                }
            }
        }
        return true;
    }
};
// ----------------------------------------

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality) {
    static MJPGWriter wri(port, timeout, quality);
    cv::Mat mat = cv::cvarrToMat(ipl);
    wri.write(mat);
    std::cout << " MJPEG-stream sent. \n";
}
// ----------------------------------------

CvCapture* get_capture_video_stream(char *path) {
    CvCapture* cap = NULL;
    try {
        cap = (CvCapture*)new cv::VideoCapture(path);
    }
    catch (...) {
        std::cout << " Error: video-stream " << path << " can't be opened! \n";
    }
	return cap;
}
// ----------------------------------------

CvCapture* get_capture_webcam(int index) {
    CvCapture* cap = NULL;
    try {
        cap = (CvCapture*)new cv::VideoCapture(index);
        //((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
    }
    catch (...) {
        std::cout << " Error: Web-camera " << index << " can't be opened! \n";
    }
    return cap;
	
}
// ----------------------------------------

IplImage* get_webcam_frame(CvCapture *cap, const CvArr *mapx, const CvArr *mapy) {
    IplImage* src = NULL;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        cv::Mat frame;

        if (cpp_cap.isOpened())
        {
            cpp_cap >> frame;
            IplImage tmp = frame;
			src = cvCloneImage(&tmp);
			cvRemap(src, src, mapx, mapy);
        }
        else {
            std::cout << " Video-stream stoped! \n";
        }
    }
    catch (...) {
        std::cout << " Video-stream stoped! \n";
    }
    return src;
}

int get_stream_fps_cpp(CvCapture *cap) {
    int fps = 25;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        fps = cpp_cap.get(CAP_PROP_FPS);
#else                        // OpenCV 2.x
        fps = cpp_cap.get(CV_CAP_PROP_FPS);
#endif
    }
    catch (...) {
        std::cout << " Can't get FPS of source videofile. For output video FPS = 25 by default. \n";
    }
    return fps;
}
// ----------------------------------------
extern "C" {
    image ipl_to_image(IplImage* src);    // image.c
}

image image_data_augmentation(IplImage* ipl, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float jitter, float dhue, float dsat, float dexp)
{
    cv::Mat img = cv::cvarrToMat(ipl);

    // crop
    cv::Rect src_rect(pleft, ptop, swidth, sheight);
    cv::Rect img_rect(cv::Point2i(0, 0), img.size());
    cv::Rect new_src_rect = src_rect & img_rect;

    cv::Rect dst_rect(cv::Point2i(std::max(0, -pleft), std::max(0, -ptop)), new_src_rect.size());

    cv::Mat cropped(cv::Size(src_rect.width, src_rect.height), img.type());
    cropped.setTo(cv::Scalar::all(0));

    img(new_src_rect).copyTo(cropped(dst_rect));

    // resize
    cv::Mat sized;
    cv::resize(cropped, sized, cv::Size(w, h), 0, 0, INTER_LINEAR);

    // flip
    if (flip) {
        cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
        sized = cropped.clone();
    }

    // HSV augmentation
    // CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB
    if (ipl->nChannels >= 3)
    {
        cv::Mat hsv_src;
        cvtColor(sized, hsv_src, CV_BGR2HSV);    // also BGR -> RGB

        std::vector<cv::Mat> hsv;
        cv::split(hsv_src, hsv);

        hsv[1] *= dsat;
        hsv[2] *= dexp;
        hsv[0] += 179 * dhue;

        cv::merge(hsv, hsv_src);

        cvtColor(hsv_src, sized, CV_HSV2RGB);    // now RGB instead of BGR
    }
    else
    {
        sized *= dexp;
    }

    // Mat -> IplImage -> image
    IplImage src = sized;
    image out = ipl_to_image(&src);

    return out;
}
class ComAsy
{
public:
	ComAsy();
	~ComAsy();
	bool InitCOM(LPCTSTR Port);
	void UninitCOM();


	bool ComWrite(LPBYTE buf, int &len);


	static unsigned int __stdcall OnRecv(void*);

private:
	HANDLE m_hCom;
	OVERLAPPED m_ovWrite;
	OVERLAPPED m_ovRead;
	OVERLAPPED m_ovWait;
	volatile bool m_IsOpen;
	HANDLE m_Thread;
};
ComAsy::ComAsy() :
	m_hCom(INVALID_HANDLE_VALUE),
	m_IsOpen(false),
	m_Thread(NULL)
{
	memset(&m_ovWait, 0, sizeof(m_ovWait));
	memset(&m_ovWrite, 0, sizeof(m_ovWrite));
	memset(&m_ovRead, 0, sizeof(m_ovRead));
}
ComAsy::~ComAsy()
{
	UninitCOM();
}
bool ComAsy::InitCOM(LPCTSTR Port)
{
	m_hCom = CreateFile(Port, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
		FILE_FLAG_OVERLAPPED | FILE_ATTRIBUTE_NORMAL,//设置异步标识
		NULL);
	if (INVALID_HANDLE_VALUE == m_hCom)
	{

		return false;
	}
	else printf("串口打开");
	SetupComm(m_hCom, 4096, 4096);//设置发送接收缓存

	DCB dcb;
	GetCommState(m_hCom, &dcb);
	dcb.DCBlength = sizeof(dcb);
	dcb.BaudRate = CBR_9600;
	dcb.StopBits = ONESTOPBIT;
	SetCommState(m_hCom, &dcb);//配置串口

	PurgeComm(m_hCom, PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR | PURGE_TXABORT);

	COMMTIMEOUTS ct;
	ct.ReadIntervalTimeout = MAXDWORD;//读取无延时，因为有WaitCommEvent等待数据
	ct.ReadTotalTimeoutConstant = 0;  //
	ct.ReadTotalTimeoutMultiplier = 0;//

	ct.WriteTotalTimeoutMultiplier = 500;
	ct.WriteTotalTimeoutConstant = 5000;

	SetCommTimeouts(m_hCom, &ct);

	//创建事件对象
	m_ovRead.hEvent = CreateEvent(NULL, false, false, NULL);
	m_ovWrite.hEvent = CreateEvent(NULL, false, false, NULL);
	m_ovWait.hEvent = CreateEvent(NULL, false, false, NULL);

	SetCommMask(m_hCom, EV_ERR | EV_RXCHAR);//设置接受事件

											//创建读取线程
	m_Thread = (HANDLE)_beginthreadex(NULL, 0, &ComAsy::OnRecv, this, 0, NULL);
	m_IsOpen = true;
	return true;
}

void ComAsy::UninitCOM()
{
	m_IsOpen = false;
	if (INVALID_HANDLE_VALUE != m_hCom)
	{
		CloseHandle(m_hCom);
		m_hCom = INVALID_HANDLE_VALUE;
	}
	if (NULL != m_ovRead.hEvent)
	{
		CloseHandle(m_ovRead.hEvent);
		m_ovRead.hEvent = NULL;
	}
	if (NULL != m_ovWrite.hEvent)
	{
		CloseHandle(m_ovWrite.hEvent);
		m_ovWrite.hEvent = NULL;
	}
	if (NULL != m_ovWait.hEvent)
	{
		CloseHandle(m_ovWait.hEvent);
		m_ovWait.hEvent = NULL;
	}
	if (NULL != m_Thread)
	{
		WaitForSingleObject(m_Thread, 5000);//µÈ´ýÏß³Ì½áÊø
		CloseHandle(m_Thread);
		m_Thread = NULL;
	}
}

bool ComAsy::ComWrite(LPBYTE buf, int &len)
{
	
	BOOL rtn = FALSE;
	DWORD WriteSize = 0;
	PurgeComm(m_hCom, PURGE_TXCLEAR | PURGE_TXABORT);
	m_ovWait.Offset = 0;
	rtn = WriteFile(m_hCom, buf, len, &WriteSize, &m_ovWrite);
	len = 0;
	if (FALSE == rtn && GetLastError() == ERROR_IO_PENDING){
		if (FALSE == ::GetOverlappedResult(m_hCom, &m_ovWrite, &WriteSize, TRUE)){
			return false;
		}
	}

	len = WriteSize;
	return rtn != FALSE;

}

unsigned int __stdcall ComAsy::OnRecv(void* LPParam){
	int receiveSign;
	ComAsy *obj = static_cast<ComAsy*>(LPParam);
	DWORD WaitEvent = 0, Bytes = 0;
	BOOL Status = FALSE;
	BYTE ReadBuf[4096];
	DWORD Error;
	COMSTAT cs = { 0 };
	while (obj->m_IsOpen){
		WaitEvent = 0;
		obj->m_ovWait.Offset = 0;
		Status = WaitCommEvent(obj->m_hCom, &WaitEvent, &obj->m_ovWait);
		if (FALSE == Status && GetLastError() == ERROR_IO_PENDING)//{
			Status = GetOverlappedResult(obj->m_hCom, &obj->m_ovWait, &Bytes, TRUE);
		}
		ClearCommError(obj->m_hCom, &Error, &cs);
		if (TRUE == Status && WaitEvent&EV_RXCHAR&& cs.cbInQue > 0){
			Bytes = 0;
			obj->m_ovRead.Offset = 0;
			memset(ReadBuf, 0, sizeof(ReadBuf));
			Status = ReadFile(obj->m_hCom, ReadBuf, sizeof(ReadBuf), &Bytes, &obj->m_ovRead);
			if (Status != FALSE)
			{
				receiveSign = ReadBuf[0] & 0xFF;
				receiveSign |= ((ReadBuf[1] << 8) & 0xFF00);
				receiveSign |= ((ReadBuf[2] << 16) & 0xFF0000);
				receiveSign |= ((ReadBuf[3] << 24) & 0xFF000000);
				receiveSign = receiveSign - 48;
				switch (receiveSign)
				{
				case 1:
					
					break;
				case 2:
					
					break;
				}
			}
			PurgeComm(obj->m_hCom, PURGE_RXCLEAR | PURGE_RXABORT);
		}
	return 0;
}

ComAsy com;

void WriteC(int sign, int x, int y) {
	//printf("x=%d,y=%d  ", x, y);
	int angle, longe;
	longe = sqrt((480 - y)*(480 - y) + (320 - x)*(320 - x));
	if (sign == 1) { 
		angle = acos((320-x) / longe) * 180 / 3.1415;
	}
	if (sign == 2) {
		angle = acos(-(320 - x) / longe) * 180 / 3.1415;
	}
	printf("longe=%d\n", longe);
	
	d[0] = 0x30 + sign;					//转动方向1lift，2right，5抓取
	d[1] = 0x2C;						//分隔位
	d[2] = 0x30 + angle / 100;			//servo1
	d[3] = 0x30 + (angle - (angle / 100) * 100) / 10;
	d[4] = 0x30 + angle % 10;
	d[5] = 0x2C;						//分隔位
	d[6] = 0x30 + longe / 100;			//servo2从 
	d[7] = 0x30 + (longe - (longe / 100) * 100) / 10;
	d[8] = 0x30 + longe % 10;
	len = strlen((char*)d);
	com.ComWrite(d, len);
	printf("%s\n", d);

}
int InitC() {
	com.InitCOM("COM3");
	return 1;
}
#endif    // OPENCV
