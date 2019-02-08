
## 파노라마 영상 만들기

- 본 문제의 가정: 사진은 직접 코드에 입력해준다.

**본코드**

```C++
#pragma warning(disable:4700);

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>


using namespace cv;


Mat MakePano(Mat *imgArray, int num);

int main() {


	Mat result;

	
	int num = 3;
	Mat imgArray[3];

	imgArray[0] = imread("C:/Users/Administrator/desktop/ex/0.jpg");
	imgArray[1] = imread("C:/Users/Administrator/desktop/ex/00.jpg");
	imgArray[2] = imread("C:/Users/Administrator/desktop/ex/000.jpg");


	result = MakePano(imgArray, num);

	imshow("res", result);

	waitKey();
	return 0;
}


Mat MakePano(Mat *imgArray, int num)
{

	Mat mainPano = imgArray[0];

	char buf[1000];

	for (int i = 1; i < num; i++)
	{
		
		Mat gray_mainImg, gray_objImg;

		cvtColor(mainPano, gray_mainImg, COLOR_RGB2GRAY);
		cvtColor(imgArray[i], gray_objImg, COLOR_RGB2GRAY);

		SiftFeatureDetector detector(0.3);

		vector<KeyPoint> point1, point2;

		Mat pointmat1, pointmat2;

		detector.detect(gray_mainImg, point1);
		detector.detect(gray_objImg, point2);


		SiftDescriptorExtractor extractor;
		Mat descriptor1, descriptor2;
		extractor.compute(gray_mainImg, point1, pointmat1);
		extractor.compute(gray_objImg, point2, pointmat2);



		FlannBasedMatcher matcher;

		vector<DMatch> matches;

		matcher.match(pointmat1, pointmat2, matches);

		double mindistance = 5000;

		double distance;

		for (int i = 0; i < pointmat1.rows; i++) {

			distance = matches[i].distance;

			if (mindistance > distance)mindistance = distance;

		}

		vector<DMatch>goodmatch;

		for (int i = 0; i < pointmat1.rows; i++) {

			if (matches[i].distance < 5 * mindistance)

				goodmatch.push_back(matches[i]);

		}

		Mat matGoodMatcges;

		sprintf(buf, "imgcheck%d.jpg", i);

		drawMatches(mainPano, point1, imgArray[i], point2, goodmatch, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow(buf, matGoodMatcges);
		sprintf(buf, "img%d.jpg", i);

		vector<Point2f> obj;

		vector<Point2f> scene;


		for (int i = 0; i < goodmatch.size(); i++) {







			obj.push_back(point1[goodmatch[i].queryIdx].pt);







			scene.push_back(point2[goodmatch[i].trainIdx].pt);







		}

		Mat homomatrix = findHomography(scene, obj, CV_RANSAC);

		Mat warp;

		warpPerspective(imgArray[i], warp, homomatrix, Size(imgArray[i].cols + mainPano.cols, imgArray[i].rows), INTER_CUBIC);



		Mat matPanorama;



		matPanorama = warp.clone();



		Mat matROI(matPanorama, Rect(0, 0, mainPano.cols, mainPano.rows));



		mainPano.copyTo(matROI);



		vector<Point> nonBlackList;



		nonBlackList.reserve(warp.rows *warp.cols);


		int max = 0;



		for (int i = 0; i < matPanorama.cols; ++i)

		{

			//   if not black: add to the list



			if (matPanorama.at<Vec3b>(matPanorama.rows / 2, i) != Vec3b(0, 0, 0))

			{

				if (max < i)max = i;

			}


			Mat img = matPanorama(Range(0, matPanorama.rows), Range(0, max));

			mainPano = img;


		}
		
		imshow(buf, mainPano);

	}

	return mainPano;
}
```


**코드 설명**

**- main 함수**

![image](https://user-images.githubusercontent.com/46413594/52459815-ac7fc480-2baa-11e9-85f4-ea3a4a0cd289.png)

main 함수에서 파노라마를 만들길 원하는 사진을 넣고 MakePano함수를 실행시켜 파노라마를 만든 후 보여주게 출력을 보여 주게 됩니다.


```C++
Mat result;
```

이 부분은 MakePano 함수를 실행 시킨 뒤 나오는 파노라마를 Mat형식으로 저장하는 변수를 선언해줍니다.


```C++
  int num = 3;
	Mat imgArray[3];

	imgArray[0] = imread("C:/Users/Administrator/desktop/ex/0.jpg");
	imgArray[1] = imread("C:/Users/Administrator/desktop/ex/00.jpg");
	imgArray[2] = imread("C:/Users/Administrator/desktop/ex/000.jpg");
```

여기에서는 파노라마를 만들기 원하는 사진의 개수를 num에 넣어 주고 그 수에 맞게 왼쪽부터 **순서대로** 이미지의 경로를 써주시면 됩니다.


```C++
result = MakePano(imgArray, num);

	imshow("res", result);
  ```
  
  imshow를 통해 result에 저장된 파노라마가 보여지게 됩니다. "res" 부분은 파노라마가 보여지게 되는 창의 제목을 지정해줍니다.
  
  
  
 **- MakePano함수**
 
 ![image](https://user-images.githubusercontent.com/46413594/52460109-1e0c4280-2bac-11e9-9ae4-36ebf0871e0b.png)
 
 함수의 인자는 처음 입력 이미지들의 경로를 담고 있는 배열 포인터와 그 이미지의 수를 받아야합니다.
 
 ![image](https://user-images.githubusercontent.com/46413594/52460402-6aa44d80-2bad-11e9-905a-602ae1a83cdf.png)

  파노라마를 만드는 이 함수가 동작하는 과정은 위처럼 왼쪽부터 오른쪽으로 하나씩 추가하며 같은점을 찾아 붙여가는 것입니다.(imgArray 배열의 0번 인덱스부터 누적시켜가며 붙여갑니다.)
  
  ```C++
  Mat mainPano = imgArray[0];
  ```
  mainPano는 처음에는 imgArray 배열의 첫번째 값을 받고 있고 붙여나가는 작업이 계속 되어가면서 누적되어 붙여진 사진들이 저장되어지는 변수입니다.
  
  
  ```C++
  char buf[1000];
  ```
  buf배열은 나중에 사진을 띄워주는 창의 이름에 번호를 매기기 위해 선언한 배열입니다.
  
  
  ```C++
  for (int i = 1; i < num; i++)
  ```
  이 배열은 함수 맨아래 반환하기 전까지 둘러싸고 있으며 총 받은 사진의 개수-1 만큼 작동하게 됩니다. (하나씩 누적하며 붙여나가므로)
  
  
  
  ```C++
  		Mat gray_mainImg, gray_objImg;

		cvtColor(mainPano, gray_mainImg, COLOR_RGB2GRAY);
		cvtColor(imgArray[i], gray_objImg, COLOR_RGB2GRAY);
  ```
 붙일 두 사진인 mainPano, imgArray[i]를 특징점찾기 용이하게 회색으로 바꿔 gray_mainImg와 gray_objImg에 각각 저장해줍니다.
 
 
 ```C++
 SiftFeatureDetector detector(0.3);

		vector<KeyPoint> point1, point2;

		Mat pointmat1, pointmat2;

		detector.detect(gray_mainImg, point1);
		detector.detect(gray_objImg, point2);


		SiftDescriptorExtractor extractor;
		Mat descriptor1, descriptor2;
		extractor.compute(gray_mainImg, point1, pointmat1);
		extractor.compute(gray_objImg, point2, pointmat2);
 ```
SiftFeatureDetector를 통해 Sift에서의 detector를 선언해줍니다. dectector안의 숫자는 임계값(threshold)입니다.
point 1, 2 는 회색 영상에서 코너점이나 엣지같은 식별이 용이한 특징점을 detector.detect를 통해 찾아 선택해 각각 저장하는 변수입니다.
SiftDescriptorExtractor를 통해서 Sift에서의 extractor를 선언해줍니다. 이는 아래 그림과 같이 detector를 통해 선택한 특징점들을 4x4블록으로 나누고 각 블록에 속한 gradient 방향과 크기에 대한 히스토그램을 구한 후 이 히스토그램 bin 값들을 일렬로 쭉 연결한 128차원 벡터를 pointmat 1,2에 저장합니다.(4x4 블록에 8개의 gradient 방향이라 128차원)

 ![image](https://user-images.githubusercontent.com/46413594/52461182-3cc10800-2bb1-11e9-9de2-4f5a90097198.png)
 
 
 
![image](https://user-images.githubusercontent.com/46413594/52461420-44cd7780-2bb2-11e9-851a-93383944e782.png)

```C++
FlannBasedMatcher matcher;

		vector<DMatch> matches;

		matcher.match(pointmat1, pointmat2, matches);
```
FlanBasedMatcher를 통해 pointmat 1,2에 저장되어 있는 점들을 매치시키고 그 매치된 정보를 matches 변수에 저장합니다.


```C++
double mindistance = 5000;

		double distance;

		for (int i = 0; i < pointmat1.rows; i++) {

			distance = matches[i].distance;

			if (mindistance > distance)mindistance = distance;

		}
```
매치시킨 점들 사이의 최소 가로 거리를 mindistance를 구합니다.


```C++
  vector<DMatch>goodmatch;

	for (int i = 0; i < pointmat1.rows; i++) {

			if (matches[i].distance < 5 * mindistance)

				goodmatch.push_back(matches[i]);

		}
```
mindistance값의 5배가 되는 위치에 위치한 점들을 제외하고 goodmatch변수에 넣습니다.(파노라마를 만들 때 거리가 멀어 붙이지 않아도 되는 점 ,outlier제거)


```C++
		Mat matGoodMatcges;

		sprintf(buf, "imgcheck%d.jpg", i);
		drawMatches(mainPano, point1, imgArray[i], point2, goodmatch, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow(buf, matGoodMatcges);
 ```
 goodmatch변수에 있는 점들을 이어 창을 띄워 보여줍니다. sprintf함수는 큰 반복문이 돌 때마다 띄워주는 창의 이름을 바꾸기 위해 사용했습니다.
 
 
 
 
 ![image](https://user-images.githubusercontent.com/46413594/52461922-6cbdda80-2bb4-11e9-91c5-ca648a1eb490.png)
 
 ```C++
 		vector<Point2f> obj;
		vector<Point2f> scene;

		for (int i = 0; i < goodmatch.size(); i++) {
			obj.push_back(point1[goodmatch[i].queryIdx].pt);
			scene.push_back(point2[goodmatch[i].trainIdx].pt);
		}
```
obj와 scene에 goodmatch변수에 있는 점들을 저장



```C++
Mat homomatrix = findHomography(scene, obj, CV_RANSAC);
```
두 사진의 공통 특징점사이의 변형정도를 알 수 있는 H matrix를 구합니다.


```C++
Mat warp;
warpPerspective(imgArray[i], warp, homomatrix, Size(imgArray[i].cols + mainPano.cols, imgArray[i].rows), INTER_CUBIC);
Mat matPanorama;
		matPanorama = warp.clone();
```
붙일 이미지를 구한 H matrix를 적용해 변형시킵니다. 그리고 그값을 matPanorama에 붙여 넣습니다.


```C++
Mat matROI(matPanorama, Rect(0, 0, mainPano.cols, mainPano.rows));
		mainPano.copyTo(matROI);
```
mainPano와 matPanorama를 붙여줍니다.


```C++
int max = 0;

		for (int i = 0; i < matPanorama.cols; ++i)
		{
			if (matPanorama.at<Vec3b>(matPanorama.rows / 2, i) != Vec3b(0, 0, 0))
			{
				if (max < i)max = i;
			}
			Mat img = matPanorama(Range(0, matPanorama.rows), Range(0, max));
			mainPano = img;

		}
 ```
 
 붙여주게되면 검은색 부분이 많이 남게 되는데 이 부분을 일정부분 잘라내어 줍니다.
 
 
 ```C++
 		sprintf(buf, "img%d.jpg", i);
		imshow(buf, mainPano);
 ```
 만들어진 사진을 보여줍니다.


