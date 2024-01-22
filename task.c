#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N 20000

// Создание глобальных переменных
int it, itmax = 100;
int tasks_num, task_size, extra_task;
double maxeps = 0.1e-7;
double **A, **B;

// Объявление используемых функций
void init();
double relax();
void verify();

int main(int an, char **as) {
    // Выделение памяти для матриц
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)calloc(N, sizeof(double));
    }

    // Вычисление оптимального количества и размера задач:
    tasks_num = N / 100 + 80; // Вычисление оптимального общего количества задач используя найденную зависимость от N
    task_size = N / tasks_num; extra_task = N % tasks_num; // Переменные для вычисления размера задач

    // Время старта параллельной программы
    double start_time = omp_get_wtime(); 

    init(); // Заполнение матрицы 

    // Обновление значений переменных для итераций метода релаксации (так как первая и последняя строки не обрабатываются)
    task_size = (N - 2) / tasks_num; extra_task = (N - 2) % tasks_num;
    // Релаксация может завершиться раньше, если будет достигнута необходимая точность
    for (it = 1; it <= itmax; it++) if (relax() < maxeps) break;

    // Обновление значений переменных для верификации
    task_size = N / tasks_num; extra_task = N % tasks_num;
    verify(); // Проверка результата

    // Вывод результата и времени работы
    printf("  Run time OpenMP: %fs\n  P = %d\n", omp_get_wtime() - start_time, omp_get_max_threads());

    // Освобождение памяти
    for (int i = 0; i < N; i++) {free(A[i]); free(B[i]);}
    free(A); free(B);

    return 0;
}

void init() {
    #pragma omp parallel
    #pragma omp single // Выполнение секции только одним потоком, который разделит задачи
    {
        // Переменные определяющие диапазоны строк в задачах 
        int start = 0, end = 0, dif = 0;

        // Равномерное распределение задач
        for (int p = 0; p < tasks_num; p++) {
            start = p * task_size + dif;
            end = (p + 1) * task_size + ((p < extra_task) ? dif++ + 1 : dif);
            
            // Создание задач для инициализации матрицы
            #pragma omp task 
            {
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < N; j++) {
                        if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
                        else A[i][j] = (1. + i + j);
                    }
                }
            }
        }
    }
}


double relax() {
    double eps = 0.; // Общая для всех потоков переменная
    double eps_local = 0.; // Локальная переменная для каждого потока
    #pragma omp parallel shared(eps) firstprivate(eps_local)
    #pragma omp single // Выполнение секции только одним потоком, который разделит задачи
    {
        // Переменные определяющие диапазоны строк в задачах
        int start = 0, end = 0, dif = 1;

        // Равномерное распределение задач
        if (it % 2) {
            for (int p = 0; p < tasks_num; p++) {
                start = p * task_size + dif;
                end = (p + 1) * task_size + ((p < extra_task) ? dif++ + 1 : dif);

                // Создание задач для заполнения матрицы актуальными значениями
                #pragma omp task 
                {
                    for (int i = start; i < end; i++) {
                        for (int j = 1; j <= N - 2; j++) {
                            B[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                            eps_local = Max(eps_local, fabs(A[i][j] - B[i][j])); // Наибольшее локальное значение внутри потока
                        }
                    }

                    #pragma omp critical // Необходима атомарность, чтобы избежать потери данных
                    eps = Max(eps, eps_local); // Наибольшее значение для всех потоков
                }
            }
        } else {
            for (int p = 0; p < tasks_num; p++) {
                start = p * task_size + dif;
                end = (p + 1) * task_size + ((p < extra_task) ? dif++ + 1 : dif);

                // Создание задач для заполнения матрицы актуальными значениями
                #pragma omp task 
                {
                    for (int i = start; i < end; i++) {
                        for (int j = 1; j <= N - 2; j++) {
                            A[i][j] = (B[i - 1][j] + B[i + 1][j] + B[i][j - 1] + B[i][j + 1]) / 4.;
                            eps_local = Max(eps_local, fabs(A[i][j] - B[i][j])); // Наибольшее локальное значение внутри потока
                        }
                    }

                    #pragma omp critical // Необходима атомарность, чтобы избежать потери данных
                    eps = Max(eps, eps_local); // Наибольшее значение для всех потоков
                }
            }
        }
    }


    return eps;
}


void verify() {
    double s = 0.0; // Общая для всех потоков переменная
    double s_local = 0.; // Локальная переменная для каждого потока
    #pragma omp parallel shared(s) firstprivate(s_local)
    #pragma omp single // Выполнение секции только одним потоком, который раздедит задачи
    {
        // Переменные определяющие диапазоны строк в задачах
        int start = 0, end = 0, dif = 0;

        // Равномерное распределение задач
        if (it % 2) {
            for (int p = 0; p < tasks_num; p++) {
                start = p * task_size + dif;
                end = (p + 1) * task_size + ((p < extra_task) ? dif++ + 1 : dif);
                
                // Создание задач для проверки результата
                #pragma omp task
                {
                    for (int i = start; i < end; i++) {
                        for (int j = 0; j < N; j++) {
                            // Сумма внутри каждого потока
                            s_local += A[i][j] * (i + 1) * (j + 1) / (N * N);
                        }
                    }

                    #pragma omp atomic // Необходима атомарность, чтобы избежать потери данных
                    s += s_local; // Суммарное значение для всех потоков
                }
            }
        } else {
            for (int p = 0; p < tasks_num; p++) {
                start = p * task_size + dif;
                end = (p + 1) * task_size + ((p < extra_task) ? dif++ + 1 : dif);
                
                // Создание задач для проверки результата
                #pragma omp task
                {
                    for (int i = start; i < end; i++) {
                        for (int j = 0; j < N; j++) {
                            // Сумма внутри каждого потока
                            s_local += B[i][j] * (i + 1) * (j + 1) / (N * N);
                        }
                    }

                    #pragma omp atomic // Необходима атомарность, чтобы избежать потери данных
                    s += s_local; // Суммарное значение для всех потоков
                }
            }
        }
    }

    printf("  S = %f\n", s);
}