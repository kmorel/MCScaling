#include <MCScalingConfig.h>

#include <mpi.h>

#include <stdlib.h>
#include <stdio.h>

#include <fstream>
#include <iostream>

#include <vector>

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/GenerateInterpolatedCells.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/MarchingCubes.h>

#include <dax/cont/DeviceAdapterSerial.h>

#define LOAD_DATA 1
//#define REMOVE_UNUSED_POINTS 1

class MarchingCubesExampleMPIError : public dax::cont::ErrorControl
{
public:
  MarchingCubesExampleMPIError(const char *file,
                           dax::Id line,
                           const char *funcall,
                           int errorCode)
  {
    std::stringstream message;
    message << file << ":" << line
            << ": Failed MPI call (" << funcall << ")\n";
    char errormsg[MPI_MAX_ERROR_STRING];
    int dummy;
    MPI_Error_string(errorCode, errormsg, &dummy);
    message << errormsg;
    this->SetMessage(message.str());
  }
};

#define MPICall(funcall) \
  { \
  int __my_result = funcall; \
  if (__my_result != MPI_SUCCESS) \
    { \
    throw MarchingCubesExampleMPIError(__FILE__, __LINE__, #funcall, __my_result); \
    } \
  }

class MarchingCubesExample
{
private:

  typedef dax::cont::ArrayContainerControlTagBasic Container;
  typedef dax::cont::DeviceAdapterTagSerial DeviceAdapter;

  typedef dax::cont::UniformGrid<DeviceAdapter> UniformGridType;
  typedef dax::cont::UnstructuredGrid<
      dax::CellTagTriangle,Container,Container,DeviceAdapter>
      UnstructuredGridType;

  typedef dax::cont::ArrayHandle<
      dax::Scalar, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleScalar;
  typedef dax::cont::ArrayHandle<
      dax::Vector3, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleVector;

  int Rank;

  FILE *LogFile;

  void LogTime(MPI_Comm comm,
               std::string description,
               int trial,
               float value)
  {
    int numProcesses;
    MPI_Comm_size(comm, &numProcesses);

    float average;
    MPI_Reduce(&value, &average, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    average /= numProcesses;

    float minimum;
    MPI_Reduce(&value, &minimum, 1, MPI_FLOAT, MPI_MIN, 0, comm);

    float maximum;
    MPI_Reduce(&value, &maximum, 1, MPI_FLOAT, MPI_MAX, 0, comm);

    if (this->Rank == 0)
      {
      std::cout << description << ": " << minimum
                << " - " << average << " - " << maximum << std::endl;
      fprintf(this->LogFile, "%s,%d,%d,%g,%g,%g\n",
              description.c_str(), numProcesses, trial,
              maximum, average, minimum);
      }
  }

  static UniformGridType CreateUniformGrid(int piece, int numPieces)
  {
    dax::Extent3 cellExtent;
    cellExtent.Min = dax::make_Id3(0, 0, 0);
    cellExtent.Max = dax::make_Id3(GRID_SIZE-2, GRID_SIZE-2, GRID_SIZE-2);

    int lowPiece = 0;
    int highPiece = numPieces;
    int axis = 0;
    while ((highPiece - lowPiece) > 1)
      {
      int midPiece = (highPiece + lowPiece)/2;
      dax::Id midCell = (cellExtent.Max[axis] + cellExtent.Min[axis] + 1)/2;
      if (piece < midPiece)
        {
        cellExtent.Max[axis] = midCell-1;
        highPiece = midPiece;
        }
      else
        {
        cellExtent.Min[axis] = midCell;
        lowPiece = midPiece;
        }
      axis++;
      if (axis > 2) { axis = 0; }
      }
    assert(lowPiece == piece);
    assert(highPiece == piece+1);

    dax::Extent3 pointExtent;
    pointExtent.Min = cellExtent.Min;
    pointExtent.Max = cellExtent.Max + dax::make_Id3(1, 1, 1);

    UniformGridType grid;
    grid.SetExtent(pointExtent);
    return grid;
  }

  static void ReadSupernovaData(const UniformGridType &grid,
                                MPI_Comm comm,
                                std::vector<dax::Scalar> &buffer)
  {
    assert(sizeof(float) == sizeof(dax::Scalar));

    MPI_File fd;
    MPICall(MPI_File_open(comm,
                          const_cast<char*>(DATA_FILE),
                          MPI_MODE_RDONLY,
                          MPI_INFO_NULL,
                          &fd));

    dax::Extent3 extent = grid.GetExtent();
    int arrayOfSizes[3];
    int arrayOfSubSizes[3];
    int arrayOfStarts[3];
    for (int i = 0; i < 3; i++)
      {
      arrayOfSizes[i] = GRID_SIZE;
      arrayOfSubSizes[i] = extent.Max[i] - extent.Min[i] + 1;
      arrayOfStarts[i] = extent.Min[i];
      }

    // Create a view in MPIIO
    MPI_Datatype view;
    MPICall(MPI_Type_create_subarray(3,
                                     arrayOfSizes,
                                     arrayOfSubSizes,
                                     arrayOfStarts,
                                     MPI_ORDER_FORTRAN,
                                     MPI_FLOAT,
                                     &view));
    MPICall(MPI_Type_commit(&view));
    MPICall(MPI_File_set_view(fd,
                              0,
                              MPI_FLOAT,
                              view,
                              const_cast<char *>("native"),
                              MPI_INFO_NULL));
    MPICall(MPI_Type_free(&view));

    // Figure out how many floats to read.
    int length = arrayOfSubSizes[0]*arrayOfSubSizes[1]*arrayOfSubSizes[2];

    buffer.resize(length);
    MPICall(MPI_File_read_all(fd,
                              &buffer.at(0),
                              length,
                              MPI_FLOAT,
                              MPI_STATUS_IGNORE));

    MPICall(MPI_File_close(&fd));
  }

  static void CreateMagnitudeField(const UniformGridType &grid,
                                   ArrayHandleScalar &data)
  {
    dax::cont::Scheduler<DeviceAdapter> scheduler;
    scheduler.Invoke(dax::worklet::Magnitude(),
                     grid.GetPointCoordinates(),
                     data);
  }

  void RunMarchingCubes(const UniformGridType &grid,
                        const ArrayHandleScalar &inArray,
                        MPI_Comm comm,
                        int trial)
  {
    MPI_Barrier(comm);
    dax::cont::Timer<DeviceAdapter> timer;
    typedef dax::cont::ArrayHandle<
        dax::Id, DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG, DeviceAdapter>
        ClassifyResultType;
    typedef dax::cont::GenerateInterpolatedCells<
        dax::worklet::MarchingCubesTopology,ClassifyResultType>
        GenerateTopologyType;

    dax::cont::Scheduler<DeviceAdapter> scheduler;

    // Run classify algorithm (determine how many cells are passed).
#ifdef LOAD_DATA
//    const dax::Scalar LOW_SCALAR = 0.07;
//    const dax::Scalar HIGH_SCALAR = 1.0;
    const dax::Scalar ISOVALUE = 0.07;
#else // LOAD_DATA
//    const dax::Scalar LOW_SCALAR = 50.0;
//    const dax::Scalar HIGH_SCALAR = 200.0;
    const dax::Scalar ISOVALUE = 250.5;
#endif // LOAD_DATA
    ClassifyResultType classificationArray;
//    dax::cont::Timer<DeviceAdapter> classifyTimer;
    scheduler.Invoke(dax::worklet::MarchingCubesClassify(ISOVALUE),
                     grid,
                     inArray,
                     classificationArray);
//    dax::Scalar elapsedClassify = classifyTimer.GetElapsedTime();

    // Build marching cubes topology.
    GenerateTopologyType generateTopology(
          classificationArray,
          dax::worklet::MarchingCubesTopology(ISOVALUE));
#ifndef REMOVE_UNUSED_POINTS
    generateTopology.SetRemoveDuplicatePoints(false);
#endif
    UnstructuredGridType outGrid;
    scheduler.Invoke(generateTopology, grid, outGrid, inArray);

#ifdef REMOVE_UNUSED_POINTS
    // Compact scalar array to new topology.
    ArrayHandleScalar outArray;
    resolveTopology.CompactPointField(inArray, outArray);
#endif

    // Copy grid information to host, if necessary.
    outGrid.GetCellConnections().GetPortalConstControl();
    outGrid.GetPointCoordinates().GetPortalConstControl();
#ifdef REMOVE_UNUSED_POINTS
    outArray.GetPortalConstControl();
#endif

    dax::Scalar time = timer.GetElapsedTime();

    int localNumCells = outGrid.GetNumberOfCells();
    int globalNumCells;
    MPI_Reduce(&localNumCells, &globalNumCells, 1, MPI_INT, MPI_SUM, 0, comm);
    if (this->Rank == 0)
      {
      std::cout << "Number of output cells: " << globalNumCells << std::endl;
      }

    this->LogTime(comm, "Dax-MPI", trial, time);
//    this->LogTime(comm, "classify", trial, elapsedClassify);
//    this->LogTime(comm, "scan", trial, dax::cont::scheduling::ScanTime);
//    this->LogTime(comm, "fill-index", trial, dax::cont::scheduling::FillIndexTime);
//    this->LogTime(comm, "upper-bounds", trial, dax::cont::scheduling::UpperBoundsTime);
//    this->LogTime(comm, "worklet", trial, dax::cont::scheduling::WorkletTime);
//    // this->LogTime(cont, "remove-duplicate-points", trial, dax::cont::scheduling::RemoveDuplicatePointsTime);
  }

  void TryWithComm(MPI_Comm comm)
  {
    int numProcesses;
    MPI_Comm_size(comm, &numProcesses);

    UniformGridType grid = this->CreateUniformGrid(this->Rank, numProcesses);

#ifdef LOAD_DATA
    if (this->Rank == 0) { std::cout << "Reading data..." << std::endl; }
    std::vector<dax::Scalar> buffer;
    ReadSupernovaData(grid, comm, buffer);
    if (this->Rank == 0) { std::cout << "Data read." << std::endl; }
#endif // LOAD_DATA

    for (int trial = 0; trial < NUM_TRIALS; trial++)
      {
#ifdef LOAD_DATA
      ArrayHandleScalar inArray =
          dax::cont::make_ArrayHandle(buffer, Container(), DeviceAdapter());
#else // LOAD_DATA
      if (this->Rank == 0) { std::cout << "Creating data" << std::endl; }
      ArrayHandleScalar inArray;
      CreateMagnitudeField(grid, inArray);
#endif // LOAD_DATA
      assert(grid.GetNumberOfPoints() == inArray.GetNumberOfValues());

      if (this->Rank == 0)
        {
        std::cout << "Computing Contour, " << numProcesses << " processes..."
                  << std::endl;
        }
      this->RunMarchingCubes(grid, inArray, comm, trial);
      }
  }

  void TryEachCommSize()
  {
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    if (numProcesses == 1)
      {
      std::cout << "*** WARNING: Only one process in parallel job. ***"
                << std::endl;
      std::cout << "Did you run this program with mpirun or mpiexec?"
                << std::endl;
      }

    for (int groupSize = 1; groupSize <= numProcesses; groupSize++)
      {
      bool inGroup = this->Rank < groupSize;
      MPI_Comm comm;
      MPI_Comm_split(MPI_COMM_WORLD,
                     static_cast<int>(inGroup),
                     this->Rank,
                     &comm);

      if (inGroup)
        {
        this->TryWithComm(comm);
        }

      MPI_Comm_free(&comm);
      MPI_Barrier(MPI_COMM_WORLD);
      }
  }

public:
  MarchingCubesExample(int argc, char *argv[])
  {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &this->Rank);

    this->LogFile = NULL;
    if (this->Rank == 0)
      {
      this->LogFile = fopen("MCScalingMPI.csv", "w");
      assert(this->LogFile != NULL);
      fprintf(this->LogFile, "Implementation,Threads,Trial,Seconds\n");
      }
  }
  ~MarchingCubesExample()
  {
    if (this->Rank == 0)
      {
      fclose(this->LogFile);
      this->LogFile = NULL;
      }
    MPI_Finalize();
  }

  int Run()
  {
    try
      {
      this->TryEachCommSize();
      }
    catch (dax::cont::Error error)
      {
      std::cout << "Caught Dax error: " << std::endl
                << error.GetMessage() << std::endl;
      return 1;
      }
    catch (...)
      {
      std::cout << "Undefined error caught." << std::endl;
      return 1;
      }

    return 0;
  }

private:
  MarchingCubesExample(const MarchingCubesExample &); // Not implemented.
  void operator=(const MarchingCubesExample &); // Not implemented.
};

int main(int argc, char *argv[])
{
  return MarchingCubesExample(argc, argv).Run();
}

