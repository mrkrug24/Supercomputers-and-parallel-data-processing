#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define Min(a,b) ((a)<(b)?(a):(b))
#define N 20000

// Создание глобальных переменных
double s, eps, maxeps = 0.1e-7;
double **A_part, **B_part, *up, *down;
int start, end, task_size, extra_rows, rank, size, useful_proc, it, itmax = 100;

// Объявление используемых функций
void init();
double relax();
void verify();

int main(int argc, char **argv){
    // Инициализация процессов
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Время старта параллельной программы
    double start_time = MPI_Wtime(); 

    // Равномерное распределение задач между процессами
    task_size = N / size; 
    extra_rows = N % size;
    start = rank * task_size + (rank < extra_rows ? rank : extra_rows);
    end = start + task_size + (rank < extra_rows ? 1 : 0) - 1;
    task_size = end - start + 1;

    // Контроль процессов, которым не достались задачи (если N < size)
    useful_proc = Min(N, size); 

    // Выделение памяти для интервала строк матриц внутри каждого процесса
    // Процессы будут хранить и обрабатывать данные в своих непересекающихся диапазонах строк
    // Общение будет происходить с помощью отправки сообщений соседним процессам, в которых будут содержаться граничные между интервалами строки (up, down)
    A_part = (double **)malloc(task_size * sizeof(double *));
    B_part = (double **)malloc(task_size * sizeof(double *));
    up = (double *)malloc(N * sizeof(double));
    down = (double *)malloc(N * sizeof(double));
    
    for (int i = 0; i < task_size; i++) {
        A_part[i] = (double *)malloc(N * sizeof(double));
        B_part[i] = (double *)calloc(N, sizeof(double));
    }

    init(); // Заполнение каждым процессом выделенные ему строки 
    // Релаксация может завершиться раньше, если будет достигнута необходимая точность
    for (it = 1; it <= itmax; it++) if (relax() < maxeps) break; // Итерации метода релаксации
    verify(); // Проверка результата

    // Главный процесс выведет результат и времени работы
    if (rank == 0) printf("  S = %f\n  Run time MPI: %fs\n  P = %d\n", s, (MPI_Wtime() - start_time), size);

    // Освобождение памяти
    for (int i = 0; i < task_size; i++) {free(A_part[i]); free(B_part[i]);}
    free(A_part); free(B_part);
    free(up); free(down);
    
    MPI_Finalize();

	return 0;
}

// Инициализация выделенных строк для каждого процесса, учитывая сдвиг относительно начала (start)
void init() {
    for (int i = 0; i < task_size; i++) {
        for (int j = 0; j < N; j++) {
            if (i + start == 0 || i + start == N - 1 || j == 0 || j == N - 1) A_part[i][j] = 0.;
            else A_part[i][j] = (1. + i + start + j);
        }
    }
}

// Итерация метода релаксации
double relax() {
    double eps_local = 0.;

    // В зависимости от четности итерации обновляем значения элементов матрицы A или B
    if (it % 2) {
        // Если процесс не первый и не последний, то у него 2 соседа (сверху и снизу)
        // Ему нужно отправить каждому по 1 граничной строке и получить от них аналогичные
        if (rank > 0 && rank < useful_proc - 1) {
            if (rank % 2) {
                MPI_Sendrecv(A_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(A_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(A_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(A_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        // Если процесс первый, то у него только один сосед (снизу) -> аналогичные действия
        } else if (rank == 0 && useful_proc != 1) {
            MPI_Sendrecv(A_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Если процесс последний, то у него только один сосед (сверху) -> аналогичные действия
        } else if (rank == useful_proc - 1 && useful_proc != 1) {
            MPI_Sendrecv(A_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < task_size; i++) {
            for (int j = 1; j < N - 1; j++) {
                // Обработка внутренних строк
                if (i > 0 && i < task_size - 1) {
                    B_part[i][j] = (A_part[i - 1][j] + A_part[i + 1][j] + A_part[i][j - 1] + A_part[i][j + 1]) / 4.;
                    eps_local = Max(eps_local, fabs(A_part[i][j] - B_part[i][j]));
                } 
                
                // Обработка первой строки 
                else if (i == 0 && task_size != 1 && rank != 0) {
                    B_part[i][j] = (A_part[i][j - 1] + A_part[i][j + 1] + A_part[i + 1][j] + up[j]) / 4.;
                    eps_local = Max(eps_local, fabs(A_part[i][j] - B_part[i][j]));
                }

                // Обработка последней строки
                else if (i == task_size - 1 && task_size != 1 && rank != useful_proc - 1) {
                    B_part[i][j] = (A_part[i][j - 1] + A_part[i][j + 1] + A_part[i - 1][j] + down[j]) / 4.;
                    eps_local = Max(eps_local, fabs(A_part[i][j] - B_part[i][j]));
                }

                // Если процессу досталась только 1 строка, нужно использовать up и down
                else if (i == 0 && task_size == 1 && rank != 0 && rank != useful_proc - 1) {
                    B_part[i][j] = (A_part[i][j - 1] + A_part[i][j + 1] + up[j] + down[j]) / 4.;
                    eps_local = Max(eps_local, fabs(A_part[i][j] - B_part[i][j]));
                }
            }
        }
    } else {
        // Если процесс не первый и не последний, то у него 2 соседа (сверху и снизу)
        // Ему нужно отправить каждому по 1 граничной строке и получить от них аналогичные
        if (rank > 0 && rank < useful_proc - 1) {
            if (rank % 2) {
                MPI_Sendrecv(B_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(B_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(B_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(B_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        // Если процесс первый, то у него только один сосед (снизу) -> аналогичные действия
        } else if (rank == 0 && useful_proc != 1) {
            MPI_Sendrecv(B_part[task_size - 1], N, MPI_DOUBLE, rank + 1, 0, down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Если процесс последний, то у него только один сосед (сверху) -> аналогичные действия
        } else if (rank == useful_proc - 1 && useful_proc != 1) {
            MPI_Sendrecv(B_part[0], N, MPI_DOUBLE, rank - 1, 0, up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < task_size; i++) {
            for (int j = 1; j < N - 1; j++) {
                // Обработка внутренних строк
                if (i > 0 && i < task_size - 1) {
                    A_part[i][j] = (B_part[i - 1][j] + B_part[i + 1][j] + B_part[i][j - 1] + B_part[i][j + 1]) / 4.;
                    eps_local = Max(eps_local, fabs(B_part[i][j] - A_part[i][j]));
                } 
                
                // Обработка первой строки 
                else if (i == 0 && task_size != 1 && rank != 0) {
                    A_part[i][j] = (B_part[i][j - 1] + B_part[i][j + 1] + B_part[i + 1][j] + up[j]) / 4.;
                    eps_local = Max(eps_local, fabs(B_part[i][j] - A_part[i][j]));
                }

                // Обработка последней строки
                else if (i == task_size - 1 && task_size != 1 && rank != useful_proc - 1) {
                    A_part[i][j] = (B_part[i][j - 1] + B_part[i][j + 1] + B_part[i - 1][j] + down[j]) / 4.;
                    eps_local = Max(eps_local, fabs(B_part[i][j] - A_part[i][j]));
                }

                // Если процессу досталась только 1 строка, нужно использовать up и down
                else if (i == 0 && task_size == 1 && rank != 0 && rank != useful_proc - 1) {
                    A_part[i][j] = (B_part[i][j - 1] + B_part[i][j + 1] + up[j] + down[j]) / 4.;
                    eps_local = Max(eps_local, fabs(B_part[i][j] - A_part[i][j]));
                }
            }
        }
    }

    // Поиск наибольшего значения eps_local и отправление его всем процессам
    MPI_Allreduce(&eps_local, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return eps;
}

// Проверка результата
void verify() {
    double s_local = 0.; // Локальная переменная для каждого процесса
    // В зависимости от четности отработанного количества итерации 
    // используем матрицу A или B для вычисления результата
    if (it % 2) {
        for (int i = 0; i < task_size; i++) {
            for (int j = 0; j < N; j++) {
                s_local = s_local + A_part[i][j] * (i + 1 + start) * (j + 1) / (N * N);
            }
        }
    } else {
        for (int i = 0; i < task_size; i++) {
            for (int j = 0; j < N; j++) {
                s_local = s_local + B_part[i][j] * (i + 1 + start) * (j + 1) / (N * N);
            }
        }
    }
    
    // Отправление главному процессу суммы всех значений s_local от кажого процесса
    MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}