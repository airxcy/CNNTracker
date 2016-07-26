//
//  RefCntMem.hpp
//  ITF_Inegrated
//
//  Created by Kun Wang on 9/22/2015.
//  Modified from https://github.com/davidstutz/flow-io-opencv
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef REFCNTMEM_H_
#define REFCNTMEM_H_

struct CRefCntMemPtr         // shared component of reference counted memory
{
    void *m_memory;         // allocated memory
    int m_refCnt;           // reference count
    int m_nBytes;           // number of bytes
    bool m_deleteWhenDone;  // delete memory when ref-count drops to 0
    void (*m_delFn)(void *ptr); // optional delete function
};

// reference-counted memory allocator
class CRefCntMem {
public:
    CRefCntMem(void);           // default constructor
    CRefCntMem(const CRefCntMem& ref);  // copy constructor
    ~CRefCntMem(void);          // destructor
    CRefCntMem& operator=(const CRefCntMem& ref);  // assignment

    void ReAllocate(int nBytes, void *memory, bool deleteWhenDone,
                    void (*deleteFunction)(void *ptr) = 0);
        // allocate/deallocate memory
    int NBytes(void);           // number of stored bytes
    bool InBounds(int i);       // check if index is in bounds
    void* Memory(void);         // pointer to allocated memory
private:
    void DecrementCount(void);  // decrement the reference count and delete if done
    void IncrementCount(void);  // increment the reference count
    CRefCntMemPtr *m_ptr;       // shared reference-counted memory pointer
};

#endif  // REFCNTMEM_H_