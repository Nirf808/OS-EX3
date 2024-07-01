//
// Created by nirfi on 26/06/2024.
//

#include "MapReduceFramework.h"
#include "Barrier.h"
#include <pthread.h>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <map>
#include <unistd.h>

/**************************************
 *             CONSTANTS              *
 **************************************/
#define LOCK_ERR_MSG "[[Barrier]] error on pthread_mutex_lock"
#define UNLOCK_ERR_MSG


struct ReduceContext
{
    OutputVec &outputVec;
    pthread_mutex_t *reduceMutex;
};

typedef struct ThreadContext
{
    int job_id;
    int thread_id;
    const MapReduceClient *client;
    const InputVec *inputVec;
    std::atomic<int> *atomic_counter;
    std::atomic<int> *progress;
    Barrier *barrier;
    std::vector<IntermediatePair> *threads_vector;
    std::vector<IntermediateVec> *shuffle_vector;
    ReduceContext *reduceContext;
    int threads_amount;
} ThreadContext;

struct Job
{
    bool is_finished;
    stage_t stage;

    std::atomic<int> *progress;
    size_t stage_size;
    pthread_t *threads;
    int num_threads;
    pthread_mutex_t *reduceMutex;
    ReduceContext *reduce_context;
    std::vector<IntermediateVec> *shuffle_vector;
    std::vector<IntermediatePair> *threads_vectors;
    Barrier *barrier;
    std::atomic<int> *atomic_counter;
    JobHandle job_handle;
    ThreadContext *threadContexts;
};

int num_jobs = 0;
std::map<int, Job *> jobs;

bool
pairComparator (const std::pair<K2 *, V2 *> a, const std::pair<K2 *, V2 *> b)
{
  return *(a.first) < *(b.first);
}

void reduceVectorBuilder (
    size_t total_size,
    std::vector<IntermediatePair> *threads_vectors,
    std::vector<IntermediateVec> *v,
    int num_threads,
    std::atomic<int> *progress,
    std::atomic<int> *atomic_counter);

void shuffle_stage (ThreadContext *tc)
{
  tc->atomic_counter->store (-1);
  tc->progress->store (0);
  size_t total_size = 0;
  for (int i = 0; i < tc->threads_amount; i++)
    {
      total_size += tc->threads_vector[i].size ();
    }
  jobs[tc->job_id]->stage = SHUFFLE_STAGE;
  jobs[tc->job_id]->stage_size = total_size;

  reduceVectorBuilder (total_size, tc->threads_vector, tc->shuffle_vector, tc->threads_amount,
                       tc->progress, tc->atomic_counter);

  tc->progress->store (0);
  jobs[tc->job_id]->stage = REDUCE_STAGE;
  jobs[tc->job_id]->stage_size = tc->shuffle_vector->size ();
}

void *
thread_wrapper (
    void *arg
)
{
  auto *tc = (ThreadContext *) arg;
  std::vector<std::pair<K2 *, V2 *>> *vector = tc->threads_vector
      + tc->thread_id;
  int old_value = (*(tc->atomic_counter))++;
  while (old_value < tc->inputVec->size ())
    {
      tc->client->map (tc->inputVec->at(old_value).first,
                      tc->inputVec->at(old_value)
          .second, vector);
      old_value = (*(tc->atomic_counter))++;
      (*(tc->progress))++;
    }
  std::sort (vector->begin (), vector->end (), pairComparator);
  (tc->barrier)->barrier (); // this one waits for all the threads to finish
  // map+sort phase

  if (tc->thread_id == 0){
    // move shuffle here
    shuffle_stage(tc);
  }

  (tc->barrier)->barrier (); // this one waits for the shuffle phase to finish

  //reduce
  old_value = (*(tc->atomic_counter))--;
  while (old_value >= 0)
    {
      const IntermediateVec reduce_param = (*(tc->shuffle_vector))[old_value];
      tc->client->reduce (&reduce_param, tc->reduceContext);
      old_value = (*(tc->atomic_counter))--;
      (*(tc->progress))++;
    }
  return nullptr;
}

void waitForJob (JobHandle job)
{
  Job *structJob = jobs[*(int *) job];
  if (!structJob->is_finished)
    {
      for (int i = 0; i < structJob->num_threads; i++)
        {
          pthread_join (structJob->threads[i], nullptr);
        }
      structJob->is_finished = true;
    }
}

void closeJobHandle (JobHandle job)
{
  waitForJob (job);
  int job_id = *(int *) job;
  Job *job_to_close = jobs[job_id];
  delete(job_to_close->progress);
  delete[](job_to_close->threads);
  delete(job_to_close->reduceMutex);
  delete(job_to_close->reduce_context);
  delete(job_to_close->shuffle_vector);
  delete[](job_to_close->threads_vectors);
  delete(job_to_close->barrier);
  delete(job_to_close->atomic_counter);
  delete[](job_to_close->threadContexts);
  delete((int *)(job_to_close->job_handle));

  job_to_close->progress = nullptr;
  job_to_close->threads = nullptr;
  job_to_close->reduceMutex = nullptr;
  job_to_close->reduce_context = nullptr;
  job_to_close->shuffle_vector = nullptr;
  job_to_close->threads_vectors = nullptr;
  job_to_close->barrier = nullptr;
  job_to_close->atomic_counter = nullptr;
  job_to_close->job_handle = nullptr;
  job_to_close->threadContexts = nullptr;
  delete(job_to_close);
  jobs[job_id] = nullptr;
}

bool checkEqualityK2 (K2 *k1, K2 *k2)
{
  return !(*k1 < *k2 || *k2 < *k1);
}

void reduceVectorBuilder (
    size_t total_size,
    std::vector<IntermediatePair> *threads_vectors,
    std::vector<IntermediateVec> *v,
    int num_threads,
    std::atomic<int> *progress,
    std::atomic<int> *atomic_counter)
{
  while (total_size > 0)
    {
      K2 *currentMax = nullptr;
      for (int i = 0; i < num_threads; i++)
        {
          if ((!threads_vectors[i].empty ()))
            {
              if (currentMax == nullptr || (*currentMax <
                                            *(threads_vectors[i].back ()
                                                .first)))
                {
                  {
                    currentMax = threads_vectors[i].back ().first;
                  }
                }
            }
        }
      IntermediateVec tmp;
      for (int j = 0; j < num_threads; j++)
        {
          while ((!threads_vectors[j].empty ()) && checkEqualityK2
              (threads_vectors[j].back ().first, currentMax))
            {
              tmp.push_back (threads_vectors[j].back ());
              threads_vectors[j].pop_back ();
              total_size--;
              (*progress)++;
            }
        }
      v->push_back (tmp);
      (*atomic_counter)++;
    }
}

void emit2 (K2 *key, V2 *value, void *context)
{
  IntermediatePair pair = std::pair<K2 *, V2 *> (key, value);
  auto *vector_context =
      (std::vector<IntermediatePair> *) context;
  vector_context->push_back (pair);
}

/*
 * The function receives as input output element (K3, V3) and context which
 * contains data structure of the thread that created the output element.
 * The function saves the output element in the context data structures
 * (output vector). In addition, the function updates the number of output
 * elements using atomic counter.
 */
void emit3 (K3 *key, V3 *value, void *context)
{
  OutputPair pair = OutputPair (key, value);
  auto *reduceContext = (ReduceContext *) context;
  auto *reduceMutex = reduceContext->reduceMutex;
  if (pthread_mutex_lock (reduceMutex) != 0)
    {
      fprintf (stderr, LOCK_ERR_MSG);
      exit (1);
    }
  reduceContext->outputVec.push_back (pair);
  if (pthread_mutex_unlock (reduceMutex) != 0)
    {
      fprintf (stderr, "[[Barrier]] error on pthread_mutex_unlock");
      exit (1);
    }
}

/*
 * This function gets a JobHandle and updates the state of the job into
 * the given JobState struct.
 */
void getJobState (JobHandle job, JobState *state)
{
  int *jobInt = (int *) job;
  Job *currentJob = jobs[*jobInt];
  state->stage = currentJob->stage;
  state->percentage = ((float) currentJob->progress->load () / (float)
      currentJob->stage_size) * 100;
}

/*
 * This function starts running the MapReduce algorithm (with several
 * threads) and returns a JobHandle.
 */
JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel) {
  // current job initializations
  jobs[num_jobs] = new Job();
  int job_id = num_jobs++;
  jobs[job_id]->stage = UNDEFINED_STAGE;
  jobs[job_id]->num_threads = multiThreadLevel;
  jobs[job_id]->is_finished = false;
  auto *progress = new std::atomic<int> (0);
  jobs[job_id]->progress = progress;
  auto *reduceMutex = new pthread_mutex_t ();
  pthread_mutex_init (reduceMutex, nullptr);

  auto *reduce_context = new ReduceContext {
      outputVec, reduceMutex
  };

  auto *shuffle_vector = new std::vector<IntermediateVec> ();
  auto *threads_vectors = new std::vector<IntermediatePair>[multiThreadLevel];
  auto *barrier = new Barrier (multiThreadLevel);
  auto *atomic_counter = new std::atomic<int> (0);
  auto *threads = new pthread_t [multiThreadLevel];
//  auto *threadContexts = new ThreadContext[multiThreadLevel];

  JobHandle job_handle = new int (job_id);

  jobs[job_id]->stage = MAP_STAGE;
  jobs[job_id]->stage_size = inputVec.size ();
  jobs[job_id]->reduceMutex = reduceMutex;
  jobs[job_id]->reduce_context = reduce_context;
  jobs[job_id]->shuffle_vector = shuffle_vector;
  jobs[job_id]->threads_vectors = threads_vectors;
  jobs[job_id]->barrier = barrier;
  jobs[job_id]->atomic_counter = atomic_counter;
  jobs[job_id]->job_handle = job_handle;
  jobs[job_id]->threadContexts = new ThreadContext [multiThreadLevel];



  for (int i = 0; i < multiThreadLevel; i++) {
      jobs[job_id]->threadContexts[i] = ThreadContext{ job_id, i, &client, &inputVec,
          atomic_counter, progress, barrier, threads_vectors, shuffle_vector,
          reduce_context, multiThreadLevel};
      pthread_create (threads + i, nullptr, thread_wrapper, jobs[job_id]->threadContexts
      + i);
    }

  jobs[job_id]->threads = threads;
  return job_handle;
}



