
/*
 * vsib.c -- Linux character driver for VSIbrute PCI board. 
 *
 * [indent settings: -i3 -ts3 -br -bad -bap -c2 -lps -fca -bbb -nut -l100 -npcs -ppi3]
 *
 * $Log: vsib.c,v $
 * Revision 1.5  2008/09/10 10:40:17  phi196
 * Update for 2.6.26 kernel
 *
 * Revision 1.4  2007/10/08 04:17:03  phi196
 * Added kernel version check
 *
 * Revision 1.3  2006/12/06 00:37:36  phi196
 * Updates for 2.6 kernel
 *
 * Revision 1.2.10.4  2006/08/15 02:08:16  phi196
 * Finns 2.6 version
 *
 * Revision 1.24  2006/07/12 07:14:25  jwagner
 * added Chris' VSIB_GET_BIGBUF_SIZE ioctl
 *
 * Revision 1.23  2006/07/12 07:01:46  jwagner
 * Chris' file_operations and some yucky USE_DEVFS_OR_SYSFS cleanup
 *
 * Revision 1.22  2006/07/12 06:13:29  jwagner
 * took into account deprecated and replaced MODULE_PARM
 *
 * Revision 1.21  2006/04/21 10:51:28  jwagner
 * fixed devfs_handle_t different declaration in 2.4 vs 2.6
 *
 * Revision 1.20  2006/04/21 09:44:05  jwagner
 * do request_region a second time if fails
 *
 * Revision 1.19  2006/04/21 05:58:56  jwagner
 * autoselection between devfs, old sysfs, new sysfs
 *
 * Revision 1.18  2006/04/18 09:10:51  jwagner
 * modified for post-2.6.12 sysfs
 *
 * Revision 1.17  2006/04/04 11:15:32  jwagner
 * dumped deprecated devfs for k2.6, sysfs with udev works fine
 *
 * Revision 1.16  2006/04/04 09:56:27  jwagner
 * corrected wrong warning for pci_read_config_dword()
 *
 * Revision 1.15  2006/04/04 08:41:50  jwagner
 * added devfs for 2.4 as well
 *
 * Revision 1.14  2006/04/04 07:01:50  jwagner
 * added compile option to automatically insert itself into /dev/vsib (USE_DEVFS_OR_SYSFS)
 *
 * Revision 1.13  2006/03/30 12:38:03  jwagner
 * corrected missing indentation of two #define's that for some odd reason caused compile error
 *
 * Revision 1.12  2006/03/29 11:27:14  jwagner
 * added module usage counter keeping for kernel 2.6
 *
 * Revision 1.11  2006/03/22 11:47:10  jwagner
 * Amazing, now it compiles for kernel 2.6 too! Test still pending. Removed unused system interrupt disable/enable macros and calls. Reformatted code and comments. Removed deprecated check_region. Converted pcibios_xxx funcs to new pci funcs.
 *
 * Revision 1.9  2005/01/19 07:19:02  amn
 * Dec-2004 JIVE software formatter changes; vsib.c seek support.
 *
 * Revision 1.8  2002/09/02 16:39:47  amn
 * Test version suitable for 50MHz/2 test pattern recording, for pb/JB.
 *
 * Revision 1.7  2002/08/09 11:26:56  amn
 * Jul-2002 first fringes Dwingeloo test version.
 *
 * Revision 1.5  2002/06/14 13:00:26  amn
 * Dwingeloo test trip version.
 *
 * Revision 1.4  2002/03/25 15:18:51  amn
 * First chained DMA ring buffer.
 *
 * Revision 1.3  2002/03/21 11:24:50  amn
 * Chain of DMA descriptors, 1000 descrs, 10 1-sec/1k descrs, rest reused memory.
 *
 * Revision 1.2  2002/02/27 14:33:28  amn
 * Changed Log line to be on next line than the comment start characters.
 *
 * Revision 1.1  2002/02/27 14:24:38  amn
 * Initial version.
 *
 */

/*
 * Copyright (C) 2001--2002 Ari Mujunen, Ari.Mujunen@hut.fi
 * 
 * This program is free software; you can redistribute it and/or modify it 
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2, or (at your option) any
 * later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along 
 * with this program; if not, write to the Free Software Foundation, Inc., 
 * 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.  
 */

#ifdef MODULE
#   include <linux/module.h>
#   include <linux/version.h>
// #include <linux/modversions.h>
#   if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
#      define VSIBMOD_INC_USE_COUNT   try_module_get(THIS_MODULE)
#      define VSIBMOD_DEC_USE_COUNT   module_put(THIS_MODULE)
#   else
#      define VSIBMOD_INC_USE_COUNT MOD_INC_USE_COUNT
#      define VSIBMOD_DEC_USE_COUNT MOD_DEC_USE_COUNT
#   endif
#else
#   define VSIBMOD_INC_USE_COUNT
#   define VSIBMOD_DEC_USE_COUNT
#endif

#   if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20))
#include <linux/autoconf.h>
#else
#include <linux/config.h>
#endif

#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
#   include <linux/init.h>
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,26)
#define CLASS_DEV_DESTROY(class, devt) \
   device_destroy(class, devt)
#else
#define CLASS_DEV_DESTROY(class, devt) \
   class_device_destroy(class, devt)
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,26)
#define CLASS_DEV_CREATE(class, devt, device, name) \
   device_create(class, device, devt, name)
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,15)
#define CLASS_DEV_CREATE(class, devt, device, name) \
   class_device_create(class, NULL, devt, device, name)
#else
#define CLASS_DEV_CREATE(class, devt, device, name) \
   class_device_create(class, devt, device, name)
#endif

#include <linux/types.h>
#include <linux/fs.h>
#include <linux/mm.h>   /* for 'verify_area()' */
#include <linux/errno.h>   /* for '-EBUSY' and other error codes */
#include <asm/uaccess.h>   /* for 'copy_to_user()' et al */
#include <linux/pci.h>
#include <asm/io.h>  /* for 'inp/out' and 'virt_to_bus()' */
#include <linux/ioport.h>  /* for 'check/request_region()' */
#include <linux/bigphysarea.h>   /* for getting large DMA buffer */
#include <asm/div64.h>  /* for getting 64-bit '%' op */
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
#if   (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,20))
#include <linux/devfs_fs_kernel.h>
#else
#endif
#endif

#if (LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,16))
#  define VSIB_MODULE_INT_PARAM(n, v)      static int n = v; MODULE_PARM(n, "i")
#  define VSIB_MODULE_UINT_PARAM(n, v)     static unsigned int n = v; MODULE_PARM(n, "i")
#else
#  define VSIB_MODULE_INT_PARAM(n, v)   static int n = v; module_param(n, int, 0)
#  define VSIB_MODULE_UINT_PARAM(n, v)  static unsigned int n = v; module_param(n, uint, 0)
#endif

#include "vsib_ioctl.h"

// vsib module configuration and debug code flags
#define DEBUG_ALL_WRITES      0
#define DEBUG_DESCRINIT       0
#define DEBUG_DMA_BUFFER_ALLOCATION 0
#define NORMAL_LINEAR_DESCRS  1
#define USE_ABORT             1
#define USE_ABORT_OR_STOP     1
#define USE_DEVFS_OR_SYSFS    1
#define VSIB_DMACHAINING      0
#define VSIB_KLOG             0
#define VSIB_MONITORBUFS      1
#define VSIB_VERIFYAREA       0
#define DEBUG_ALL_READS       0

#if USE_DEVFS_OR_SYSFS
#   if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,0))  
       // 'udev' package is required for automagic /dev/vsib entry
#      if (LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,12))
#         define USE_SYSFS_26_pre12    1    // use class_simple_*
#      else
#         define USE_SYSFS_26_post12   1    // use class_device_*
#      endif
#   else
      // default to devfs for older kernels
#     define USE_DEVFS_2x              1
#   endif
#endif

/*
 * Exported / local declarations
 */
static loff_t lseek_vsib(struct file *filep, loff_t offset, int orig);
static ssize_t write_vsib(struct file *filep, const char *buf, size_t count, loff_t * offset);
static ssize_t read_vsib(struct file *filep, char *buf, size_t count, loff_t * offset);
static int ioctl_vsib(struct inode *node, struct file *filep, unsigned int cmd,
                      unsigned long int arg);
static int open_vsib(struct inode *node, struct file *filep);
static int release_vsib(struct inode *node, struct file *filep);


/*
 * Constants, driver name and number of consecutive I/O addresses reqd. 
 */
#define MYNAME "vsib"
#define MYCLASS "misc"
#define NUMOFIOADDR     (256)
#define NUMOFCMDADDR    (4)
#define VSIB_DEVICE_ID  (0x5406)

#ifdef MODULE_LICENSE
MODULE_AUTHOR("Ari.Mujunen@hut.fi");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("A char driver for Metsahovi VSIB I/O PCI board");
MODULE_SUPPORTED_DEVICE("Metsahovi VSIB I/O PCI board");
#endif

/*
 * Global, 'insmod'-externally settable parameters. 
 */
VSIB_MODULE_UINT_PARAM(fix, VSIB_MODE_STOP);
MODULE_PARM_DESC(fix, "VSIB command word value to be loaded after init");

/* during module loading; can be used to quickly test command words. */
int io = 0;                     /* zero meaning "auto"detect/default */
char *plxaddr = NULL;           /* memory-mapped PLX conf area */
unsigned int *cmdaddr = NULL;   /* memory-mapped command register */
unsigned int localaddr = 0x60000000;   /* VSIbrute on-board DMA local address */

/* for all DMA transfers (kept constant, all data from the same address */

VSIB_MODULE_INT_PARAM(descrs, 1000);   /* number of DMA descriptors and data memory * blocks */
MODULE_PARM_DESC(descrs, "Number of scatter/gather DMA descriptors");
int descrbufsize = 0;           /* size of DMA descr buffer, aligned to 16 * bytes */
int allocated_descrbufsize = 0; /* larger area needed to provide aligned */

VSIB_MODULE_INT_PARAM(bigbufsize, 0);
MODULE_PARM_DESC(bigbufsize, "Size of the secondary large ring buffer");

/*
 * The struct of file operation function pointers 
 * which we will register with VFS. 
 * xxx: can probably eliminate all NULLs by using named fields 
 */
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))

 struct file_operations vsib_fops = {
  .owner = THIS_MODULE,
  .llseek = lseek_vsib,
  .read = read_vsib,
  .write = write_vsib,
  .ioctl = ioctl_vsib,
  .open = open_vsib,
  .release = release_vsib,
};

#else

/* Kernel 2.6 and later, newer syntax */
static struct file_operations vsib_fops = {
 owner:THIS_MODULE,
 llseek:lseek_vsib,
 read:read_vsib,
 write:write_vsib,
 ioctl:ioctl_vsib,
 open:open_vsib,
 release:release_vsib
};
#endif

/* My own major (xxx: dynamically allocated) device number. */
static int vsib_major;

/* xxx: */
static char *vsib_descrbuf;     /* the whole 'allocated_descrbufsize' buffer; */

// normal virt. address, not specially aligned 
typedef unsigned int tAddress32;
static tAddress32 vsib_descrbuf_bus;   /* bus address of start of bigphysbuf */

/*
* (to ensure that alignment calculations are within the allocated buffer 
* static char *vsib_descr; 
* // the 128kB-aligned part inside whole dmabuf 
*/
static tAddress32 vsib_descr;   /* bus addresses for DMA chip */
static tAddress32 vsib_descrend; /* bus addresses for DMA chip */

#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
// VSIB PCI device data required for enumeration/finding
// Note: static, load only one instance of vsib module!
static struct pci_dev *p_vsib_dev = NULL;
#endif

/* for sysfs: */
#if USE_DEVFS_OR_SYSFS
#   ifdef USE_SYSFS_26_pre12
struct class_simple *pClassSimpleVSIB;
#   endif
#   ifdef USE_SYSFS_26_post12
struct class *pClassVSIB;
//struct class_device *pClassDeviceVSIB;
#   endif
#   ifdef USE_DEVFS_2x
#      if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,0))
static struct devfs_handle_t devfs_handle; // newer kernel headers
#      else
static devfs_handle_t devfs_handle; // older kernel headers
#      endif
#   endif
#endif

/*
 * PLX DMA descriptors, as accepted by hardware. 
 */
typedef struct sPLXDMADescr
{
   tAddress32 baddr;            /* bus addr ptr to data area of this * descriptor */
   tAddress32 laddr;            /* PLX local bus address value for this * transfer block */
   unsigned int size;           /* number of bytes to transfer */
   tAddress32 next;             /* bus addr ptr to next descr; typically next in table */
} tPLXDMADescr;
static tPLXDMADescr *vsib_dd;   /* virt addr ptr to DMA descr table */

/*
 * This is initialized to the start of aligned area 'vsib_descr'. 
 * xxx: 
 */
static char *vsib_bigbuf;       /* the large buffer; normal virt. address */
static tAddress32 vsib_big;     /* bus addresses for DMA chip */

/* The point in bigbuf which is incremented by read()/write() code. */
static tAddress32 vsib_big_first_unread;

/* Nicer name for the same thing for 'write()' routine use. */
#define vsib_big_first_unwritten vsib_big_first_unread

/* The point in bigbuf which is incremented by PLX DMA hardware. */
static tAddress32 vsib_big_first_vacant;

#define vsib_big_DMA_point vsib_big_first_vacant
static int vsib_big_first_write_call_after_open;

/* Macros for safely disabling interrupts during ISA/PCI DMA chip access. 
 * Disables /all/ interrupts, screws up multiprocessor systems.
 * This needs a local stack-allocated 'unsigned long int flags'.  */
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
#   define INT_OFF save_flags(flags); cli()
#   define INT_ON  restore_flags(flags)
#else

#endif

/* Our timer hook routine.  It gets called 100 times per second
 * and it empties the DMA buffer into another larger secondary 
 * ring buffer to prevent sleep overflows in 'reax13'. */

/*Linux kernel (sched.c) timer hook pointer. */
// extern void (*do_timer_hook)(); 

/* Saved copy of previous existing timer hook (from the prev extern var). */
// static void (*prev_timer_hook)() = NULL;

/* Keeping max values of fill degrees of both DMA and big buffers. */
static int vsib_max_dma_used = 0;
static int vsib_max_big_used = 0;   /* for read() ring buffer full detect */
static int vsib_max_big_emptied = 0;   /* for write() ring buffer empty * detect */

/*
 * PLX chip definitions. 
 */

/* Hardware defines the descriptors as four 4-byte words. */
#define PLX_DMA_DESCR_SIZE 16

/* PLX DMA cmd/status registers are only one byte each. */
#define PLX_DMA_CH0_COMMAND_STATUS 0xa8
#define PLX_DMA_CH1_COMMAND_STATUS 0xa9
#define PLX_DMA_CS_ENABLE 0x01
#define PLX_DMA_CS_START 0x02
#define PLX_DMA_CS_ABORT 0x04
#define PLX_DMA_CS_CLEAR_INT 0x08
#define PLX_DMA_CS_DONE 0x10
#define PLX_DMA_CH0_getStatus (readb(plxaddr + PLX_DMA_CH0_COMMAND_STATUS))
#define PLX_DMA_CH0_isDone (PLX_DMA_CH0_getStatus & PLX_DMA_CS_DONE)
#define PLX_DMA_CH0_putCommand(x) writeb((x), plxaddr + PLX_DMA_CH0_COMMAND_STATUS)

/* Following PLX chip DMA progress by reading DMA address registers. */
#define PLX_DMA_CH0_PCI_ADDR 0x84

/*
 * PLX_DMA_CH0_getPciAddr is not very useful, since PLX chip doesn't 
 * update the readback value of this register during one block of DMA. 
 * Instead, the descriptor pointer register gets updated every time PLX 
 * loads in a new descriptor.  (4 LSB are flags, remove them in reading.) 
 */
#define PLX_DMA_CH0_DESCR_ADDR 0x90
#define PLX_DMA_CH0_getDescrAddr ((readl(plxaddr + PLX_DMA_CH0_DESCR_ADDR)) & 0xfffffff0)

/* Non-scatter-gather test mode.  Test/debug only. */
#define SINGLEBLOCK 0

/* For debugging PCI status register. */
unsigned char bus, devfn;

/*
 * Getting the "DMA progress point" in bigbuf. 
 */
static tAddress32
getDMAPoint(void)
{
   tAddress32 descrAddr;
   int descrNum;

   /* 
    * The idea is to safely take a copy of a "pessimistic" value of 
    * DMA transfer pointer, i.e. the address of first byte in big buffer 
    * which may be still in DMA progress. 
    */
#if 0
   /* 
    * This code would be applicable also in cases when the 
    * 'vsib_big_first_vacant' counter magically changes e.g. in 
    * an interrupt routine. 
    */
   unsigned long int flags;

   INT_OFF;
   current_big_ctr = vsib_big_first_vacant;
   INT_ON;
#endif
   /* 
    * With PLX we can just take the DMA descriptor pointer 
    * from PLX register and assume PLX is transferring 
    * this particular DMA block---the previous descr has been
    * transferred. 
    * Thus the first "vacant" (i.e. "still pending") address is 
    * the starting address of currently running DMA block. 
    */
   descrAddr = PLX_DMA_CH0_getDescrAddr;
   descrNum = (descrAddr - vsib_descr) / sizeof(tPLXDMADescr);
   /* Back up one descriptor... xxx */
   if (descrNum == 0) {
      descrNum = descrs - 1;
   }
   else {
      descrNum--;
   }
   if ((descrAddr == 0) || (descrNum < 0) || (descrNum > descrs)) {
#if SINGLEBLOCK
      /* In singleblock test mode never uses descriptors; this fails always. */
#else
      printk(KERN_INFO MYNAME ": not DMAing yet, descr addr = %08X, dnum = %d\n", descrAddr,
             descrNum);
#endif
      /* xxx: return value is unsigned, cannot return -1 or similar */
      return (0);
   }
   /* The byte number of first still pending DMA transfer byte. */
   return (vsib_dd[descrNum].baddr - vsib_big);
}  /* getDMAPoint */

/*
 * Read routine that transfers accumulated stuff from VSIbrute DMA buffer. 
 */
static ssize_t
read_vsib(
            /* struct inode *node, */
            struct file *filep, char *buf, size_t count, loff_t * offset)
{
   int written = 0;

#if VSIB_VERIFYAREA
   /* 
    * Apparently new copy_to_user() checks this. 
    * Check that we can write to user buffer area. 
    */
   if (verify_area(VERIFY_WRITE, buf, count) == -EFAULT) {
      return (-EFAULT);
   }
#endif

   /* If the board was opened for writing, cannot read from it. */
   if ((filep->f_flags & O_ACCMODE) == O_WRONLY) {
      return (-EINVAL);
   }
#if VSIB_DMACHAINING /* special DMA chaining tests */
   {
      char *d;
      int i;
      unsigned char debugStatus;
      static int prevDescr00 = -1;
      int descr00;

      /* Get the kernel virtual address of start of DMA buffer. */
      d = vsib_bigbuf;  /* DMA hw. transfers phys. bus addresses at * vsib_big */
      if (count > bigbufsize) {
         count = bigbufsize;
      }

      descr00 = -1;
      /* If DMA not done, return zero bytes to indicate "wait for more". */
      if (!(debugStatus = PLX_DMA_CH0_isDone)) {
#   if SINGLEBLOCK
         printk(KERN_INFO MYNAME ": not done, DMA ch0 status = %02X\n", (int) debugStatus);
         written = 0;
         goto lExit;
#   else
         tAddress32 descrAddr = PLX_DMA_CH0_getDescrAddr;
         int descrNum = (descrAddr - vsib_descr) / sizeof(tPLXDMADescr);

         descr00 = descrNum / 100;
         if ((descrAddr == 0) || (descrNum < 0)
             || (descrNum > descrs)) {
            printk(KERN_INFO MYNAME
                   ": not done, DMA ch0 status = %02X, descr addr = %08X, dnum = %d\n",
                   (int) debugStatus, descrAddr, descrNum);
            descr00 = -1;
            written = 0;
            goto lExit;
         }
         if (prevDescr00 == descr00) {
            /* Still the same "second", a new "second start" is not available. */
            written = 0;
            goto lExit;
         }
         if ((descrNum % 100) == 0) {
            /* A new second, but the first 100,200... must complete first. */
            descr00 = prevDescr00;
            written = 0;
            goto lExit;
         }
         else {
            /* Get PLX PCI status (abort, parity err bits). */
            unsigned short pcist;

#      if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
            pcibios_read_config_word(bus, devfn, PCI_STATUS, &pcist);
#      else
            pci_read_config_word(p_vsib_dev, PCI_STATUS, &pcist);
#      endif

            /* A new "second start" is available, transfer 1kB. */
            d = vsib_bigbuf + (descr00 * 1024);
            if (count > 1024) {
               count = 1024;
            }
            printk(KERN_INFO MYNAME ": read(): returning descr[%d] 1kB from %08X, PCI st = %04X\n",
                   descrNum, (unsigned int) d, (unsigned int) pcist);
         }
#   endif
         /* else not SINGLEBLOCK */
      }
      else {
         printk(KERN_INFO MYNAME ": done, DMA ch0 status = %02X\n", (int) debugStatus);
      }

      /* Copy a block or a final "whole buffer", clearing it. */
      for (i = 0; i < count; i++) {
         put_user(*d, buf);   /* 'buf' is a pointer to * 'char'/byte */
         (*(d++)) = 0;
         buf++;
         written++;
      }  /* for */
    lExit:
      prevDescr00 = descr00;
      return written;
   }  /* block */
#endif /* special DMA chaining tests */

   {
      tAddress32 current_big_ctr;
      int avail;
      int transfer_now;

      /* The byte number of first still pending DMA transfer byte. */
      vsib_big_first_vacant = getDMAPoint();
      /* Monitoring the maximum usage of big buffer. */
#if VSIB_MONITORBUFS
      {
         int filled = (int) vsib_big_first_vacant - (int) vsib_big_first_unread;

         if (filled < 0) {
            filled = bigbufsize + filled;
         }
         if (filled > vsib_max_big_used) {
            vsib_max_big_used = filled;
            printk(KERN_INFO MYNAME ": big secondary ring buffer filled to %d bytes\n",
                   vsib_max_big_used);
         }
      }
#endif

      /* 
       * The following algorithm uses this "snapshot" copy of the master 
       * ring buffer counter; the counter may continue to advance. 
       */
      current_big_ctr = vsib_big_first_vacant;

#if DEBUG_ALL_READS
      printk(KERN_INFO MYNAME ": reading %d bytes, first_unread=%d, current_big_ctr=%d\n", count,
             vsib_big_first_unread, current_big_ctr);
#endif

#if 1
      if (vsib_big_first_unread <= current_big_ctr) {
         /* 
          * From first unread to current counter. 
          * (We are assuming the "big vacant" counter points 
          * to the next intended ring buffer address.) 
          * If first_unread and first_vacant are the same, 
          * we return zero words, 
          * i.e. nothing new in buffer. 
          */
         avail = current_big_ctr - vsib_big_first_unread;
      }
      else {
         /* 
          * From first unread to end of buffer, 
          * and then _in the second 'memcpy()'_, from start of buffer 
          * to first_vacant. 
          */
         avail = bigbufsize - vsib_big_first_unread;
      }  /* if */

      /* 
       * Transfer the first (or only) half, if fits in 'count'. 
       */
      if ((transfer_now = (count < avail ? count : avail))) {
         if (copy_to_user(buf, &(vsib_bigbuf[vsib_big_first_unread]), transfer_now)) {
            written = -EFAULT;
            goto lReturn;
         }
#   if DEBUG_ALL_READS
         printk(KERN_INFO MYNAME ": transferred %d bytes from first_unread=%d\n", transfer_now,
                vsib_big_first_unread);
#   endif

         vsib_big_first_unread += transfer_now;
         buf += transfer_now;
         count -= transfer_now;
         written = transfer_now;

         if (vsib_big_first_unread > (bigbufsize - 1)) { /* wrapped */
            /* 
             * Transfer the second half, if fits in the residue of
             * 'count'. Wrap back to beginning of ring buffer. 
             */

   /***assert***/ if (vsib_big_first_unread !=
                      bigbufsize) {
               printk(KERN_ERR MYNAME
                      ": assertion failed: vsib_big_first_unread (0x%08x) != bigbufsize (0x%08x)\n",
                      vsib_big_first_unread, bigbufsize);
               vsib_big_first_unread %= bigbufsize;
            }
            else {
               /* Algorithm works as designed, 'vsib_big_first_unread' wraps back to 0. */
               vsib_big_first_unread = 0;
            }
            avail = current_big_ctr - vsib_big_first_unread;
            if ((transfer_now = (count < avail ? count : avail))) {
               if (copy_to_user(buf, &(vsib_bigbuf[vsib_big_first_unread]), transfer_now)) {
                  written = -EFAULT;
                  goto lReturn;
               }
#   if DEBUG_ALL_READS
               printk(KERN_INFO MYNAME ": transferred residue %d bytes from first_unread=%d\n",
                      transfer_now, vsib_big_first_unread);
#   endif
               vsib_big_first_unread += transfer_now;
               written += transfer_now;
            }
         }  /* if wrapped */
#endif
      }  /* if initial transfer_now is > 0, i.e. there is something in buffer */
   }

 lReturn:
   return written;
}  /* read_vsib */

/*
 * The write routine puts user process data into the large ring buffer. 
 * PLX chip then takes care of DMAing it out to the VSIB output port. 
 */
static ssize_t
write_vsib(
             /* struct inode *node, */
             struct file *filep, const char *buf, size_t count, loff_t * offset)
{
   int readThisFar = 0;
   tAddress32 current_big_ctr;
   int avail;
   int transfer_now;

   /* If the board was not opened for writing only, cannot write to it. */
   if ((filep->f_flags & O_ACCMODE) != O_WRONLY) {
      return (-EINVAL);
   }

   /* The byte number of first still pending DMA transfer byte. */
   vsib_big_DMA_point = getDMAPoint();
   /* Monitoring the usage of big buffer dropping to zero. */
#if VSIB_MONITORBUFS
   {
      int emptied = (int) vsib_big_DMA_point - (int) vsib_big_first_unwritten;
      static int prevEmptied;

      if (emptied < 0) {
         emptied = bigbufsize + emptied;
      }
      if (vsib_big_first_write_call_after_open) {
         emptied = bigbufsize;
      }
      else {
         /* Ignore when empty space is decreasing. */
         if ((emptied > prevEmptied)
             && (emptied > vsib_max_big_emptied)) {
            vsib_max_big_emptied = emptied;
            printk(KERN_INFO MYNAME
                   ": big secondary ring buffer empty space increased to %d bytes\n",
                   vsib_max_big_emptied);
         }
      }
      prevEmptied = emptied;
   }
#endif

   /* 
    * The following algorithm uses this "snapshot" copy of the master 
    * ring buffer counter; the counter may continue to advance. 
    */
   current_big_ctr = vsib_big_DMA_point;

#if DEBUG_ALL_WRITES
   printk(KERN_INFO MYNAME ": writing %d bytes, first_unwritten=%d, current_big_ctr=%d\n", count,
          vsib_big_first_unwritten, current_big_ctr);
#endif

   if (vsib_big_first_write_call_after_open) {
      printk(KERN_INFO MYNAME
             ": first call after open(), writing %d bytes, first_unwritten=%d, current_big_ctr=%d\n",
             count, vsib_big_first_unwritten, current_big_ctr);
      /* 
       * For the first call only 
       * where first_unwritten and first_vacant are the same, 
       * allow the buffer to fill. 
       */
      vsib_big_first_write_call_after_open = 0;
      avail = bigbufsize;
   }
   else if (vsib_big_first_unwritten <= current_big_ctr) {
      /* 
       * Data from current counter to first unwritten (vacant) byte. 
       * Vacant space from unwritten to ctr. 
       * If first_unwritten and first_vacant are the same, 
       * the whole buffer is full of data to be transferred to VSIB. 
       */
      avail = current_big_ctr - vsib_big_first_unwritten;
   }
   else {
      /* 
       * From first unwritten to end of buffer, 
       * and then _in the second 'memcpy()'_, from start of buffer 
       * to current DMA point. 
       */
      avail = bigbufsize - vsib_big_first_unwritten;
   }  /* if */

   /* Transfer the first (or only) half, if fits in 'count'. */
   if ((transfer_now = (count < avail ? count : avail))) {
      if (copy_from_user(&(vsib_bigbuf[vsib_big_first_unwritten]), buf, transfer_now)) {
         readThisFar = -EFAULT;
         goto lReturn;
      }
#if DEBUG_ALL_WRITES
      printk(KERN_INFO MYNAME ": transferred %d bytes to first_unwritten=%d\n", transfer_now,
             vsib_big_first_unwritten);
#endif

      vsib_big_first_unwritten += transfer_now;
      buf += transfer_now;
      count -= transfer_now;
      readThisFar = transfer_now;

      if (vsib_big_first_unwritten > (bigbufsize - 1)) { /* wrapped */
         /* 
          * Transfer the second half, if fits in the residue of
          * 'count'. Wrap back to beginning of ring buffer. 
          */

      /***assert***/ if (vsib_big_first_unwritten !=
                         bigbufsize) {
            printk(KERN_ERR MYNAME
                   ": assertion failed: vsib_big_first_unwritten (0x%08x) != bigbufsize (0x%08x)\n",
                   vsib_big_first_unwritten, bigbufsize);
            vsib_big_first_unwritten %= bigbufsize;
         }
         else {
            /* Algorithm works as designed, 'vsib_big_first_unwritten' wraps back to 0. */
            vsib_big_first_unwritten = 0;
         }
         avail = current_big_ctr - vsib_big_first_unwritten;
         if ((transfer_now = (count < avail ? count : avail))) {
            if (copy_from_user(&(vsib_bigbuf[vsib_big_first_unwritten]), buf, transfer_now)) {
               readThisFar = -EFAULT;
               goto lReturn;
            }
#if DEBUG_ALL_WRITES
            printk(KERN_INFO MYNAME ": transferred residue %d bytes from first_unread=%d\n",
                   transfer_now, vsib_big_first_unwritten);
#endif
            vsib_big_first_unwritten += transfer_now;
            readThisFar += transfer_now;
         }
      }  /* if wrapped */
   }
   /* if initial transfer_now is > 0, i.e. there is something in buffer */
 lReturn:
#if VSIB_KLOG
   printk(KERN_INFO MYNAME ": bigbuf[7]=%08X\n", *((int *) &(vsib_bigbuf[7])));
#endif
   return readThisFar;
}  /* write_vsib */

/*
 * Seek routine which always forces VSIbrute to the "beginning" of the device. 
 */
static loff_t
lseek_vsib(
             /* struct inode *node, */
             struct file *filep, loff_t offset, int orig)
{
   unsigned long long p;

   switch (orig) {
   case 0:
      filep->f_pos = offset;
      break;
   case 1:
      filep->f_pos += offset;
      break;
      /* case 2: seek from end has no realistic meaning in a dynamic buffer */
   default:
      return -EINVAL;
   }  /* switch */

   /* Update the read()/write() point to "wrapped" size of bigbuf. */
   p = filep->f_pos;
   vsib_big_first_unread   /* = vsib_big_first_unwritten */
      = do_div(p, bigbufsize);
#if VSIB_KLOG
   printk(KERN_DEBUG MYNAME "lseek(): vsib_big_first_unread=0x%08X\n", vsib_big_first_unread);
#endif

#if 0
   /* Flush all accumulated data (if seeked to zero???). */
   vsib_big_first_unread = 0;
   vsib_big_first_vacant = 0;
#endif

   return (filep->f_pos);
}  /* lseek_vsib */


/*
 * 'ioctl()' which gives access to mode reg (w) and status reg (r). 
 */

static int
ioctl_vsib(struct inode *node, struct file *filep, unsigned int cmd, unsigned long int arg)
{
   switch (cmd) {
   case VSIB_SET_MODE:
      {
         writel(arg, cmdaddr);   /* memory-mapped "put" into * command register */
         break;
      }
   case VSIB_GET_STATUS:
      {
         char statusbyte = 0;   /* inb(STATUSREG); */
         unsigned char *p = (unsigned char *) arg;

#if VSIB_VERIFYAREA
         /* Check that we can write to user buffer area. */
         if (verify_area(VERIFY_WRITE, p, sizeof(unsigned char)) == -EFAULT) {
            return (-EFAULT);
         }
#endif
         put_user(statusbyte, p);   /* 'p' is a pointer to * 'char'/byte */
         break;
      }
   case VSIB_GET_DMA_RETRIES:
      {
         unsigned long int *p = (unsigned long int *) arg;

#if VSIB_VERIFYAREA
         /* Check that we can write to user buffer area. */
         if (verify_area(VERIFY_WRITE, p, sizeof(unsigned long int)) == -EFAULT) {
            return (-EFAULT);
         }
#endif
         put_user(0, p);
         break;
      }
   case VSIB_GET_BIGBUF_SIZE: 
      {
         unsigned long int *p = (unsigned long int *)arg;
         put_user(bigbufsize, p);
         break;
      }
   case VSIB_GET_BYTES_IN_BIGBUF:
      {
         unsigned long int *p = (unsigned long int *)arg;
         int filled;
         
#if VSIB_KLOG
         printk(KERN_INFO MYNAME ": DMA status = %02X\n", (int)(PLX_DMA_CH0_getStatus));
#endif
         
         /* xxx: doesn't stop when DMA reaches end of real data, */
         /* but instead wraps around... */
         vsib_big_DMA_point = getDMAPoint();
         filled = (int)vsib_big_first_unwritten - (int)vsib_big_DMA_point;
         
         if (filled < 0) {
         filled = bigbufsize + filled;
         }
         
         put_user(filled, p);
         break;         
      }
   case VSIB_RESET_DMA:
      {
         int clearMemoryBuffersAndRWPointers = 1;
         unsigned int u;
         int i;
         int debugStatus;
         unsigned int dmaDescrFlagsRW;

         /* 
          * When write()ing from mem to VSIB, we need to reset DMA 
          * when memory buffers have already been filled. 
          * (Default "clear" happens when arg==0 (for compatibility).) 
          */
         if (arg) {
            clearMemoryBuffersAndRWPointers = 0;
         }

         /* 
          * Descriptor address 4 LSBs are flags as follows: 
          * Bit 0: descriptor in PCI address space 
          */
#define DESCR_IN_PCI 0x00000001
         /* Bit 1: no end of chain */
#define DESCR_END_OF_CHAIN 0x00000002
         /* 
          * Bit 2: no interrupt after this transfer block 
          * Bit 3: local-->PCI==1 ('wr' to disk), ==0 PCI-->local ('rd' 
          * from d) 
          *
          * If the device special file has been opened for writing
          * only, then init DMA in direction of PCI-->local bus on VSIB
          * board. 
          */
         if ((filep->f_flags & O_ACCMODE) == O_WRONLY) {
            dmaDescrFlagsRW = 0x00000000; /* Bit 3: * PCI-->local==0 */
         }
         else {
            dmaDescrFlagsRW = 0x00000008; /* Bit 3: * local-->PCI==1 */
         }

         /* 
          * xxx: Any need to stop the VSIbrute board, apparently no,
          * since this disables the DMA first before reiniting. 
          * Initialize DMA, junking everything which may already be in
          * buffer. 
          * (Usually this is called when multiple read attempts return
          * no data.) 
          */

#if USE_ABORT_OR_STOP
         /* 
          * Re-init DMA channel 0. 
          * Disable DMA and abort and clear interrupts and all for DMA ch0. 
          * From manual: "Aborting when no DMA in progress causes the next DMA to abort." 
          */
         if (!(debugStatus = PLX_DMA_CH0_isDone)) {
#   if USE_ABORT
            printk(KERN_INFO MYNAME ": trying to abort DMA, status = %02X\n", (int) debugStatus);
            /* 
             * (_START bit is really required; otherwise doesn't
             * really abort and returns with done bit ==1 immediately. 
             */
            PLX_DMA_CH0_putCommand(PLX_DMA_CS_ABORT | PLX_DMA_CS_CLEAR_INT | PLX_DMA_CS_START);
#   else
            /* 
             * Chip rev. AB occasionally hangs when abort bit is set
             * from PCI. So, as a workaround we clear all '.next' pointers in
             * DMA chain. Then our DMA should stop (as long as VSIclk is
             * running...). This typically takes about 675000/4/32000000==5.3msec
             * though... 
             */
            for (i = 0; i < descrs; i++) {
               vsib_dd[i].next = 0 /* no next */  |
                  DESCR_IN_PCI | dmaDescrFlagsRW | DESCR_END_OF_CHAIN;
            }
#   endif
            /* Wait for abort/last transfer to complete, "done==1". */
            i = 0;
            while (!(debugStatus = PLX_DMA_CH0_isDone)) {
               i++;
               /* 
                * Asking for the status takes a minimum of
                * 2cycles*30ns. Completing the current block takes 2.7--12msec,
                * even more. It is not polite to busy wait so long, so with rev. 
                * AC chips we should revert back to using the real abort bit. 
                * 1000000 queries * 60ns --> about 60msec. 
                */
#   if USE_ABORT
               if (i > 100) {
#   else
               if (i > 1000000) {
#   endif
                  /* 
                   * xxx: This really happens quite easily e.g. when 
                   * non-existent DMA PCI target addresses are used by PLX chip. 
                   * xxx: Must find a stronger "master reset" for PLX. 
                   */
                  printk(KERN_ERR MYNAME ": aborting DMA failed, status = %02X\n",
                         (int) debugStatus);
                  break;   /* prevent looping * forever */
               }
            }  /* while not done */
         }  /* if DMA was in progress */

         /* Clear interrupt, cancel START bit (possibly still on). */
         PLX_DMA_CH0_putCommand(PLX_DMA_CS_CLEAR_INT);
#endif /* if use abort/stop */

         if (clearMemoryBuffersAndRWPointers) {
            /* Clear memory buffers (to aid in debugging). */
            char *d;

            d = vsib_descrbuf;   /* virt. addr */
            for (i = 0; i < descrbufsize; i++) {
               d[i] = 0;
            }
            d = vsib_bigbuf;  /* virt. addr */
            for (i = 0; i < bigbufsize; i++) {
               d[i] = 0;
            }
         }

         /* 
          * Define PCI command codes to be used on PCI bus when doing
          * DMA. The default PCI memory read code is MRL, cache line only. 
          * We always burst long regions, thus MRM, memory read
          * multiple is more appropriate and newer MBs can do it more
          * efficiently. 
          */
         {
            unsigned int old = readl(plxaddr + 0x6c);

            printk(KERN_INFO MYNAME ": CNTRL was = %08X\n", old);
            /* 
             * Change PCI read command code from power-up default 
             * MRL (memory real (cache) line) --> MRM (memory read
             * multiple). 
             */
            writel((old & 0xfffffff0) | 0x0000000c, plxaddr + 0x6c);
            /* 
             * xxx: could perhaps change PCI write command code 
             * from MW --> MWI (0xf) (memory write and invalidate) 
             * writel((old & 0xffffff00) | 0x000000fc , plxaddr+0x6c); 
             */
            old = readl(plxaddr + 0x6c);
            printk(KERN_INFO MYNAME ": CNTRL is now = %08X\n", old);
         }

         /* 
          * 32-bit, 0WS, enable bursting, hold local addr constant, 
          * Demand mode, DMA fast/slow stop mode =0, slow==BLAST used. 
          * BTERM# input is enabled, so PLX doesn't do Burst-4 but 
          * does Burst-forever; Xilinx doesn't ask for extra ADS cycles 
          * but instead keeps BTERM#==1 always. 
          */
#if SINGLEBLOCK
         writel(0x00000003 /* 0-1: 32-bit local bus */
                | 0x00000100  /* 8: local bus bursting enabled */
                /* 9: NO scatter/gather mode enabled */
                | 0x00000800  /* 11: keep local address * constant */
                | 0x00001000  /* 12: demand mode (DREQ/DACK * signals used) */
                , plxaddr + 0x80);
#else
         writel(0x00000003 /* 0-1: 32-bit local bus */
                /* 
                 * 2-5: 0 wait states 
                 * 6: TA#/READY# input not enabled 
                 */
                | 0x00000080  /* 7: BTERM# input enabled (but * not used by Xilinx) */
                | 0x00000100  /* 8: local bus bursting enabled */
                | 0x00000200  /* 9: scatter/gather mode * enabled */
                /* 10: done interrupt not enabled */
                | 0x00000800  /* 11: keep local address * constant */
                | 0x00001000  /* 12: demand mode (DREQ/DACK * signals used) */
                /* 
                 * 13: no special PCI write and invalidate 
                 * 14: EOT# pin not used 
                 * 15: slow mode termination (2 before /w BLAST) 
                 * 16: auto-zero count after transfer in descr 
                 * 17: interrupt to ==0 local, ==1 PCI int 
                 * 18: PCI DAC dual-address cycle...? not enabled, >4GB address space 
                 * 19--31: reserved 
                 */
                , plxaddr + 0x80);
#endif
         /* PCI address; physical bus address of DMA buffer. */
         writel(vsib_big, plxaddr + PLX_DMA_CH0_PCI_ADDR);
         /* Local bus address; CSxxx matches to a single/same LA 'localaddr'. */
         writel(localaddr, plxaddr + 0x88);
         /* 
          * xxx: These PCI+local addr are probably not needed at all 
          * in scatter/gather mode, they are overwritten by descr values. 
          */

#if SINGLEBLOCK
         /* 
          * Descriptor pointer; initially no descriptors/chaining/scatter, 
          * but direction Local<->PCI is determined by flags in this reg. 
          */
         writel(dmaDescrFlagsRW, plxaddr + PLX_DMA_CH0_DESCR_ADDR);
         /* Transfer byte count; use the max of bigbuf. */
         writel(bigbufsize, plxaddr + 0x8c);
#else
         /* 
          * Chain of multiple descriptors, started by the first descr
          * ptr. Descriptor pointer; init to DMA descr table. 
          * xxx: Must keep start of descr table 16 byte (4lword)
          * aligned! 
          * The PLX chip needs the descriptor pointer as PCI bus
          * physical address and our setup code needs the kernel virtual addresses. 
          */
         writel(u = ((tAddress32) virt_to_bus(vsib_dd) | DESCR_IN_PCI
                     /* Bit 0: descriptor in PCI address space Bit 1: no end of chain Bit 2: no
                        interrupt after this transfer block */
                     | dmaDescrFlagsRW /* Bit 3: * local-->PCI==1 */
                ), plxaddr + PLX_DMA_CH0_DESCR_ADDR);
         printk(KERN_INFO MYNAME ": virt ptr to 1st descr = %08X\n", ((unsigned int) vsib_dd));
         printk(KERN_INFO MYNAME ": bus  ptr to 1st descr = %08X\n", u);

         /* Init chain of descriptors, dividing bigbuf equally to all descrs. */
         {
            unsigned int blocksize = bigbufsize / descrs;   /* xxx: any limits to transfer size? */

            printk(KERN_INFO MYNAME ": calculated transfer size of each descr = %u\n", blocksize);
            /* 
             * VSIbrute apparently doesn't handle sub-32-bit-word
             * transfers. In theory PLX allows it, but requires the use of byte
             * lane enables. 
             */
            blocksize &= ~(0x00000003);   /* align to 4 * bytes */

            /* 
             * xxx: should definitely check for: 
             * assert( (blocksize * descrs) == bigbufsize ); 
             */

            printk(KERN_INFO MYNAME ": transfer size of each descr = %u\n", blocksize);

#   if NORMAL_LINEAR_DESCRS
            /* Normal "linear" descriptors. */
            for (i = 0; i < descrs; i++) {
               /* PCI hardware bus address for DMA data transfer. */
               u = vsib_dd[i].baddr = vsib_big + i * blocksize;
#      if DEBUG_DESCRINIT
               printk(KERN_INFO MYNAME ": %d. descr PCI addr = %08X\n", i, u);
#      endif
               /* 
                * PLX on-board local bus address, always the same
                * 'localaddr'. 
                */
               u = vsib_dd[i].laddr = localaddr;
#      if DEBUG_DESCRINIT
               printk(KERN_INFO MYNAME ": %d. descr loc addr = %08X\n", i, u);
#      endif
               u = vsib_dd[i].size = blocksize; /* transfer size */
#      if DEBUG_DESCRINIT
               printk(KERN_INFO MYNAME ": %d. descr tr. size = %08X\n", i, u);
#      endif
               u = vsib_dd[i].next = ((tAddress32)
                                      virt_to_bus(&(vsib_dd[i + 1])))
                  | DESCR_IN_PCI | dmaDescrFlagsRW;
#      if DEBUG_DESCRINIT
               printk(KERN_INFO MYNAME ": %d. descr next des = %08X\n", i, u);
#      endif
            }  /* for each descriptor */
            /* 
             * Fix the pointer of last descriptor to point back to
             * start of chain. 
             */
            i--;
            u = vsib_dd[i].next = ((tAddress32)
                                   virt_to_bus(&(vsib_dd[0]))) | DESCR_IN_PCI | dmaDescrFlagsRW;
#      if VSIB_DMACHAINING
            /* No chain. */
            u = vsib_dd[i].next = 0 /* no next */  | DESCR_IN_PCI |
               dmaDescrFlagsRW | DESCR_END_OF_CHAIN;
#      endif
            printk(KERN_INFO MYNAME ": last (%d) descr next des = %08X\n", i, u);
            printk(KERN_INFO MYNAME ": last (%d) descr PCI bus  = %08X\n", i, vsib_dd[i].baddr);
#   else
            /* 
             * Special DMA chaining test descriptors. 
             * Testing with small 5MB bigbuf; same 1.3MB is "reused"
             * and only small 1kB "one-second-marker-blocks" are created 
             * in the beginning of bigbuf. 
             */
            for (i = 0; i < descrs; i++) {
               /* PCI hardware bus address for DMA data transfer. */
               u = vsib_dd[i].baddr = vsib_big + 10 * 1024;
               /* PLX on-board local bus address, always the same 'localaddr'. */
               u = vsib_dd[i].laddr = localaddr;
               u = vsib_dd[i].size = 4 * 32000 * 10;  /* 1.3MB, * 1280000 * bytes */
               u = vsib_dd[i].next = ((tAddress32)
                                      virt_to_bus(&(vsib_dd[i + 1])))
                  | DESCR_IN_PCI | dmaDescrFlagsRW;
            }  /* for each descriptor */
            /* 
             * Fix the pointer of last descriptor. 
             */
            i--;
#      if VSIB_DMACHAINING
            /* No ring. */
            u = vsib_dd[i].next = 0 /* no next */  | DESCR_IN_PCI |
               dmaDescrFlagsRW | DESCR_END_OF_CHAIN;
#      else
            /* Automatic (hw) ring buffer. */
            u = vsib_dd[i].next = ((tAddress32)
                                   virt_to_bus(&(vsib_dd[0]))) | DESCR_IN_PCI | dmaDescrFlagsRW;
#      endif
            /* Fix 1-second descriptors (0, 100, 200...). */
            for (i = 0; i < 10; i++) {
               u = vsib_dd[i * 100].baddr = vsib_big + i * 1024;
               printk(KERN_INFO MYNAME ": fix descr[%d] PCI addr = %08X\n", i * 100, u);
               u = vsib_dd[i * 100].size = 1024;
               u = (vsib_dd[i * 100 + 1].size *= 2);
               u = (vsib_dd[i * 100 + 1].size -= 1024);
            }
#      if 0
            /* Move 100,200... ffff ffff marker in middle of 1024-byte buffer. */
            u = (vsib_dd[1].size -= 512);
#      endif

#   endif
            /* else not NORMAL_LINEAR_DESCRS */
         }
#endif /* else not SINGLEBLOCK */

         /* Enable and start DMA ch0. */
         PLX_DMA_CH0_putCommand(PLX_DMA_CS_ENABLE | PLX_DMA_CS_START);
         printk(KERN_INFO MYNAME ": PLX DMA ch0 started\n");

         if (clearMemoryBuffersAndRWPointers) {
            vsib_big_first_unread = 0;
            vsib_big_first_vacant = 0;
         }
         break;
      }
   case VSIB_DELAYED_STOP_DMA:
      {
         unsigned int dmaDescrFlagsRW;
         int i;

         /* 
          * If the device special file has been opened for writing
          * only, then init DMA in direction of PCI-->local bus on VSIB
          * board. 
          */
         if ((filep->f_flags & O_ACCMODE) == O_WRONLY) {
            dmaDescrFlagsRW = 0x00000000; /* Bit 3: * PCI-->local==0 */
         }
         else {
            dmaDescrFlagsRW = 0x00000008; /* Bit 3: * local-->PCI==1 */
         }

         /* 
          * Chip rev. AB occasionally hangs when abort bit is set from
          * PCI. So, as a workaround we clear all '.next' pointers in DMA
          * chain. 
          * Then our DMA should stop (as long as VSIclk is running...). 
          * This typically takes about 675000/4/32000000==5.3msec
          * though... 
          */
         for (i = 0; i < descrs; i++) {
            vsib_dd[i].next = 0 /* no next */  | DESCR_IN_PCI |
               dmaDescrFlagsRW | DESCR_END_OF_CHAIN;
         }
         break;
      }
   case VSIB_IS_DMA_DONE:
      {
         unsigned long int *p = (unsigned long int *) arg;

         /* Get the DMA ch0 status value, extract bit "done" and return it. */
         put_user(PLX_DMA_CH0_isDone, p);
         break;
      }
   default:
      {
         return -EINVAL;
      }
   }  /* switch */
   return 0;
}  /* ioctl_vsib */

/*
 * Open/allocate routine. 
 */
static int
open_vsib(struct inode *node, struct file *filep)
{
   int retval;

   /* xxx: if opened, don't allow re-open; only one user process at a time */

   VSIBMOD_INC_USE_COUNT;

   /* Init the board into stopped state. */
   (void) ioctl_vsib(node, filep, VSIB_SET_MODE, VSIB_MODE_STOP);
   /* 
    * xxx: wait until stopped/last cycle DACKed? 
    * Flush all accumulated data. 
    */
   vsib_big_first_unread = 0;
   vsib_big_first_vacant = 0;

   /* 
    * Enable first write call to put stuff in big buffer, 
    * although both head and tail of ring buffer == 0. 
    */
   vsib_big_first_write_call_after_open = 1;

   /* Reset DMA according to R/W mode in 'filep->f_flags'. */
   retval = ioctl_vsib(node, filep, VSIB_RESET_DMA, 0);

   return retval;
}  /* open_vsib */

/*
 * Close/release/deallocate routine. 
 */
static int
release_vsib(struct inode *node, struct file *filep)
{
   /* Stop the board. */
   (void) ioctl_vsib(node, filep, VSIB_SET_MODE, VSIB_MODE_STOP);
   /* 
    * xxx: wait until stopped/last cycle DACKed? 
    * Flush all accumulated data (will be done again in 'open()'). 
    */
   vsib_big_first_unread = 0;
   vsib_big_first_vacant = 0;

   VSIBMOD_DEC_USE_COUNT;
   return (0);
}  /* release_vsib */


// ---------------------------------------------------------------------------
#ifdef MODULE

/*
 * Driver initialization. 
 */

int
init_module(void)
{
   unsigned int tmp_uint = 0;
   u32 u = 0;

   /* Find VSIbrute PLX-based PCI card. */
#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
   if (!pcibios_present()) {
      printk(KERN_ERR MYNAME ": PCI BIOS not present or not accessible\n");
      return -ENODEV;
   }
   if (pcibios_find_device(PCI_VENDOR_ID_PLX, VSIB_DEVICE_ID, /* pci_index=> */ 0, &bus, &devfn)) {
      printk(KERN_ERR MYNAME ": PLX vendor=0x%04X, device id=0x%04X not found\n", PCI_VENDOR_ID_PLX,
             VSIB_DEVICE_ID);
      return -ENODEV;
   }
   pcibios_read_config_dword(bus, devfn, PCI_BASE_ADDRESS_0, &u);
#   else
   if (NULL != p_vsib_dev) {
      printk(KERN_INFO MYNAME ": current p_vsib_dev wasn't null, second module instance?");
   }
   // pci_present() : obsolete since 2.5; pci not present if search funcs return null
   // while(pci_find_device(PCI_VENDOR_ID_PLX, VSIB_DEVICE_ID, p_vsib_dev)) { 
   // configure_device(p_vsib_dev); 
   // }
   p_vsib_dev = pci_get_device(PCI_VENDOR_ID_PLX, VSIB_DEVICE_ID, NULL);
   if (NULL == p_vsib_dev) {
      printk(KERN_ERR MYNAME ": PLX vendor=0x%04X, device id=0x%04X not found\n", PCI_VENDOR_ID_PLX,
             VSIB_DEVICE_ID);
      return -ENODEV;
   }
   if (0 != pci_read_config_dword(p_vsib_dev, PCI_BASE_ADDRESS_0, &tmp_uint)) {
      printk(KERN_INFO " error in init_module(), could not read PCI_BASE_ADDRESS_0 \n");
   }
   u = tmp_uint;
#   endif

#   if VSIB_KLOG
   printk(KERN_DEBUG MYNAME ": base0=0x%08X\n", u);
#   endif

   u &= PCI_BASE_ADDRESS_MEM_MASK;
   plxaddr = ioremap(u, NUMOFIOADDR);  /* PLX configuration area */

#   if VSIB_KLOG
   printk(KERN_DEBUG MYNAME ": base0 ioremap()ped to 0x%08X (plxaddr)\n", (unsigned int) plxaddr);
#   endif

#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
   pcibios_read_config_dword(bus, devfn, PCI_BASE_ADDRESS_2, &u);
#   else
   pci_read_config_dword(p_vsib_dev, PCI_BASE_ADDRESS_2, &tmp_uint);
   u = tmp_uint;
#   endif

#   if VSIB_KLOG
   printk(KERN_DEBUG MYNAME ": base2=0x%08X\n", u);
#   endif

#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
   pcibios_read_config_dword(bus, devfn, PCI_BASE_ADDRESS_3, &u);
#   else
   pci_read_config_dword(p_vsib_dev, PCI_BASE_ADDRESS_3, &tmp_uint);
   u = tmp_uint;
#   endif

#   if VSIB_KLOG
   printk(KERN_DEBUG MYNAME ": base3=0x%08X\n", u);
#   endif

   u &= PCI_BASE_ADDRESS_MEM_MASK;
   cmdaddr = ioremap(u, NUMOFCMDADDR); /* Local bus address range 1 of * PLX */

#   if VSIB_KLOG
   printk(KERN_DEBUG MYNAME ": base3 ioremap()ped to 0x%08X (cmdaddr)\n", (unsigned int) cmdaddr);
#   endif

   /* Check PCI cache line size.  (BIOS should set a reasonable value.) */
   {
      unsigned char ub;

#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
      pcibios_read_config_byte(bus, devfn, PCI_CACHE_LINE_SIZE, &ub);
#   else
      pci_read_config_byte(p_vsib_dev, PCI_CACHE_LINE_SIZE, &ub);
#   endif

      printk(KERN_DEBUG MYNAME ": cache line size was = %i longwords\n", ub);
#   if 0
#      ifndef SMP_CACHE_BYTES
#         define SMP_CACHE_BYTES	L1_CACHE_BYTES
#      endif
      if ((ub << 2) != SMP_CACHE_BYTES) {
         printk(KERN_INFO "  PCI cache line size set incorrectly "
                "(%i bytes) by BIOS/FW, correcting to %i\n", (ub << 2), SMP_CACHE_BYTES);
         pcibios_write_config_byte(bus, devfn, PCI_CACHE_LINE_SIZE, SMP_CACHE_BYTES >> 2);
      }
#   endif
#   if 0
#      if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
      pcibios_write_config_byte(bus, devfn, PCI_CACHE_LINE_SIZE, 16);
      pcibios_read_config_byte(bus, devfn, PCI_CACHE_LINE_SIZE, &ub);
#      else
      pci_write_config_byte(p_vsib_dev, PCI_CACHE_LINE_SIZE, 16);
      pci_read_config_byte(p_vsib_dev, PCI_CACHE_LINE_SIZE, &ub);
#      endif
      printk(KERN_DEBUG MYNAME ": cache line size now = %i longwords\n", ub);
#   endif
   }

   /* 
    * Change the local starting address of local address range 1. 
    * This will be VSIbrute's command register. 
    */
   writel(0x50000001, (plxaddr + 0xf4));
   u = readl(plxaddr + 0xf4);
   printk(KERN_DEBUG MYNAME ": local address range 1 starting address changed to 0x%08X\n", u);
   /* Disable Ready input for this area. */
   writel(0x00000103, (plxaddr + 0xf8));
   u = readl(plxaddr + 0xf8);
   printk(KERN_DEBUG MYNAME ": local address range 1 region descr changed to 0x%08X\n", u);

   /* Check I/O addresses. */
   if (io <= 0) {
      /* Not set explicitly, "probe default", i.e. read PCI config. */
#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
      pcibios_read_config_dword(bus, devfn, PCI_BASE_ADDRESS_1, &u);
#   else
      pci_read_config_dword(p_vsib_dev, PCI_BASE_ADDRESS_1, &tmp_uint);
      u = tmp_uint;
#   endif

      printk(KERN_DEBUG MYNAME ": base1=0x%08X\n", u);
      u &= PCI_BASE_ADDRESS_IO_MASK;
      io = u & 0x0000ffff;
   }

   if ((bigbufsize % descrs) != 0) {
      printk(KERN_ERR MYNAME ": bigbufsize (%d) not evenly divisible with descrs (%d)\n",
             bigbufsize, descrs);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENODEV;
   }

#   if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
   if (check_region(io, NUMOFIOADDR)) {
      printk(KERN_ERR MYNAME ": unable to get I/O addresses 0x%04x--0x%04x\n", io,
             io + NUMOFIOADDR - 1);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENODEV;
   }

   /* Reserve I/O addresses after successful probing. */
   request_region(io, NUMOFIOADDR, MYNAME);
#   else
   if (request_region(io, NUMOFIOADDR, MYNAME)) {
      printk(KERN_ERR MYNAME ": request_region() unable to get I/O addresses 0x%04x--0x%04x\n", io,
             io + NUMOFIOADDR - 1);
      if (request_region(io, NUMOFIOADDR, MYNAME)) {
         printk(KERN_ERR MYNAME ": second request_region() attempt failed, too. quitting.\n");
         iounmap(plxaddr);
         iounmap(cmdaddr);
         return -ENODEV;
      } else {
         printk(KERN_INFO MYNAME ": second request_region() successful!\n");
      }
   }
#   endif

   /* 
    * Reserve a suitable DMA descriptor buffer. 
    * Assert size of the descriptor struct to be 16 bytes as PLX wants. 
    */
   if (sizeof(tPLXDMADescr) != PLX_DMA_DESCR_SIZE) {
      printk(KERN_ERR MYNAME ": gcc made sizeof(tPLXDMADescr) != %d bytes (gcc %d bytes)\n",
             PLX_DMA_DESCR_SIZE, sizeof(tPLXDMADescr));
      release_region(io, NUMOFIOADDR);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENOMEM;
   }
   /* Force multiple of 16 bytes (descriptors are that long). */
   descrbufsize &= ~(PLX_DMA_DESCR_SIZE - 1);
   /* xxx: May want to lift the arbitrary limit of no more than 128kB of descrs. */
   if ((descrbufsize <= 0) || (descrbufsize > (128 * 1024))) {
      /* 
       * Not set explicitly, "probe default". 
       *
       * By default we'll use 128kB.  (It used to be the max ISA hw DMA
       * buffer size, and this code was originally made to provide 128k-aligned 
       * 128k-buffer below 16MB, as required by ISA DMA system.) 
       */
      descrbufsize = 128 * 1024;
   }
   if (allocated_descrbufsize <= 0) {
      /* 
       * Not set explicitly, "probe default". 
       * The original reasoning was: 
       * 256kB is bound to contain 128kB which doesn't cross a 128kB
       * boundary. 
       *
       * Now we allocate "one alignment block more". 
       */
      allocated_descrbufsize = (descrbufsize + sizeof(tPLXDMADescr));
   }
   if (!(vsib_descrbuf = bigphysarea_alloc(allocated_descrbufsize))) {
      printk(KERN_ERR MYNAME ": DMA descr buffer (%d bytes) allocation failed\n",
             allocated_descrbufsize);
      release_region(io, NUMOFIOADDR);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENOMEM;
   }

   /* 
    * Orig: calculate the safe 128kB-non-boundary-crossing starting
    * address. 
    * Here we use the same ISA-DMA strategy to provide 16-byte aligned 
    * buffer / DMA descriptor table as required by PLX chip. 
    * (Four LSB of descriptor addresses are used as flag bits.) 
    */
   vsib_descrbuf_bus = vsib_descr = virt_to_bus(vsib_descrbuf);
   /* Round up to next start of 128kB block with 17 LSB address bits == 0. */
   vsib_descr = (vsib_descr & ~(PLX_DMA_DESCR_SIZE)) + PLX_DMA_DESCR_SIZE;
   /* 
    * If 'vsib_descr' is perfectly aligned, we'll end up using the latter 
    * half of the buffer, both +1 and -1 will then start at
    * vsib_descrbuf+1, 
    * and all other values will start at vsib_descrbuf+1..end. 
    */
   vsib_descrend = vsib_descr + (descrbufsize - 1);
   /* Put the table of DMA descriptors in the aligned middle of this buff. */
   vsib_dd = (tPLXDMADescr *) (bus_to_virt(vsib_descr));

   if (  /**((vsib_descrend & 0xff000000) != 0)**//* >16MB; xxx: was a ISA DMA lim. */

                        /**||**/ (vsib_descr < vsib_descrbuf_bus)
         // before buffer start
         || (vsib_descrend > (vsib_descrbuf_bus + allocated_descrbufsize - 1))
         // after buffer end
         || ((descrbufsize / sizeof(tPLXDMADescr)) < descrs)
         // too small buf for reqd # of descrptors
      ) {
      printk(KERN_ERR MYNAME ": DMA buffer at >16MB (0x%08x) or outside allocated buffer\n",
             vsib_descr);
      bigphysarea_free(vsib_descrbuf, allocated_descrbufsize);
      release_region(io, NUMOFIOADDR);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENOMEM;
   }
#   if DEBUG_DMA_BUFFER_ALLOCATION
   printk(KERN_INFO MYNAME
          ": descrbuf=0x%08x, allocated_descrbufsize=%d, bus=0x%08x--0x%08x\n",
          (tAddress32) vsib_descrbuf, allocated_descrbufsize, vsib_descr, vsib_descrend);
#   endif

   /* Reserve a large, large secondary "big" ring buffer. */
   if (bigbufsize <= 0) {
      /* Not set explicitly, "probe default". By default we'll use 5MB.  */
      bigbufsize = 5 * 1024 * 1024;
   }
   if (!(vsib_bigbuf = bigphysarea_alloc(bigbufsize))) {
      printk(KERN_ERR MYNAME ": Secondary big ring buffer (%d bytes) allocation failed\n",
             bigbufsize);
      bigphysarea_free(vsib_descrbuf, allocated_descrbufsize);
      release_region(io, NUMOFIOADDR);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return -ENOMEM;
   }
   vsib_big = virt_to_bus(vsib_bigbuf);   /* xxx: assumed auto 4-byte * align */
   vsib_big_first_unread = vsib_big_first_vacant = 0;

   /* Register this character device driver into Linux driver table. */
   if ((vsib_major = register_chrdev(0, MYNAME, &vsib_fops)) == -EBUSY) {
      printk(KERN_ERR MYNAME ": unable to get a dynamic major device number\n");
      bigphysarea_free(vsib_bigbuf, bigbufsize);
      bigphysarea_free(vsib_descrbuf, allocated_descrbufsize);
      release_region(io, NUMOFIOADDR);
      iounmap(plxaddr);
      iounmap(cmdaddr);
      return (-EIO);
   }

#   if USE_DEVFS_OR_SYSFS
#      ifdef USE_SYSFS_26_pre12

   /* Register in sysfs */
   // This will create /sys/classes/vsib/vsib/dev, and the udev daemon will then
   // automatically create /dev/vsib with correct maj+min, provided it is installed!
   pClassSimpleVSIB = class_simple_create(THIS_MODULE, MYNAME);
   if (IS_ERR(pClassSimpleVSIB)) {
      printk(KERN_ERR MYNAME ": class_simple_create() failed, but continuing anyway...\n");
   }
   else {
      class_simple_device_add(pClassSimpleVSIB, MKDEV(vsib_major, 0), NULL, MYNAME);
      printk(KERN_INFO MYNAME ": created sysfs entry (udev daemon should now create /dev/vsib)\n");
   }
#      endif
#      ifdef USE_SYSFS_26_post12
   pClassVSIB = class_create(THIS_MODULE, MYNAME);
   CLASS_DEV_CREATE(pClassVSIB, MKDEV(vsib_major, 0), NULL, MYNAME);
#      endif

#      ifdef USE_DEVFS_2x

   /* Register in devfs */
   if (devfs_register_chrdev(vsib_major, MYNAME, &vsib_fops)) {
      printk(KERN_ERR MYNAME ": could not create devfs entry\n");
   }
   else {
      devfs_handle = devfs_register(NULL, MYNAME, DEVFS_FL_DEFAULT,
                                    vsib_major, 0, S_IFCHR | S_IRUSR | S_IWUSR, &vsib_fops, NULL);
      printk(KERN_INFO MYNAME ": created devfs entry\n");
   }

#      endif
#   endif

   /* Stop the board. */
   writel(VSIB_MODE_STOP, cmdaddr);

#   if 0
   Actually deferred to 'open()' routine, when R / W mode is known.
      /* Initialize DMA. */
     (void) ioctl_vsib(NULL, NULL, VSIB_RESET_DMA, 0);
#   endif

   /* Set up our 100Hz timer hook routine.  */
   // {
   // unsigned long int flags = 0;
   // 
   // INT_OFF;
   // // prev_timer_hook = do_timer_hook; 
   // // save existing timer routine address
   // // do_timer_hook = vsib_timer_hook; 
   // INT_ON;
   // }

   /* We got the major number (and everything else). */
   printk(KERN_INFO MYNAME
          ": loaded with major=%d, I/O=0x%04x, descrbuf=0x%08x (%d bytes), "
          "bus=0x%08x--0x%08x, bigbuf=0x%08x (%d bytes), $Revision: 1.5 $\n",
          vsib_major, io, (tAddress32) vsib_descrbuf, allocated_descrbufsize, vsib_descr,
          vsib_descrend, (tAddress32) vsib_bigbuf, bigbufsize);

   /* Test-fix-write to VSIbrute command register, by default STOP. */
   writel(fix, cmdaddr);

   return 0;
}  /* init_module */

void
cleanup_module(void)
{
   // unsigned long int flags = 0;
   // 
   // INT_OFF;
   /* Remove our 100Hz timer hook routine. */
   // do_timer_hook = prev_timer_hook; restore previous timer routine addr
   // INT_ON;

#   if USE_DEVFS_OR_SYSFS
#      ifdef USE_SYSFS_26_pre12
   class_simple_device_remove(MKDEV(vsib_major, 0));
   class_simple_destroy(pClassSimpleVSIB);
#      endif
#      ifdef USE_SYSFS_26_post12
   CLASS_DEV_DESTROY(pClassVSIB, MKDEV(vsib_major, 0));
   class_destroy(pClassVSIB);
#      endif
#      ifdef USE_DEVFS_2x
   devfs_unregister(devfs_handle);
   devfs_unregister_chrdev(vsib_major, MYNAME);
#      endif
#   endif
   unregister_chrdev(vsib_major, MYNAME);
   bigphysarea_free(vsib_bigbuf, bigbufsize);

   bigphysarea_free(vsib_descrbuf, allocated_descrbufsize);
   release_region(io, NUMOFIOADDR);
   iounmap(plxaddr);
   iounmap(cmdaddr);

#   if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
   if (NULL != p_vsib_dev) {
      // release the device, decreasese the reference counter
      pci_dev_put(p_vsib_dev);
   }
#   endif

   printk(KERN_INFO MYNAME ": unloaded, max_dma=%d, max_big=%d.\n", vsib_max_dma_used,
          vsib_max_big_used);
}  /* cleanup_module */
#else /* not module, compiled into kernel */

long
vsib_init(long mem_start, long mem_end)
{
   if ((vsib_major = register_chrdev(0, MYNAME, &vsib_fops))) {
      printk(KERN_ERR MYNAME ": unable to get a dynamic major device number\n");
   }
   else {
      printk(KERN_INFO MYNAME ": detected, major device number=%d\n", vsib_major);
   }
   return mem_start;
}  /* vsib_init */

#   if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
module_init(init_module);
module_exit(cleanup_module);
#   endif

#endif /* module or compiled in */
