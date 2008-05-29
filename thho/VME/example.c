#include "tpcD32.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/vme.h> 
#include <fcntl.h>
#include <sys/uio.h>
#include <sys/vme_types.h>
#include <sys/vui.h>
#include <signal.h> 
#include <time.h>

#define RAWDATA_PRINT
#define ERROR_WITH_RAW_PRINT

#ifdef RAWDATA_PRINT
  #define RAWDATA_FILE         "./many_run_test/rawdata_log.dat"
#endif

#ifdef ERROR_WITH_RAW_PRINT
  #define ERROR_WITH_RAW_FILE         "./many_run_test/error_rawdata.dat"
#endif

int fd, fd1;
ioctl_irq_t intr;
static struct TPC_REG *tpc[MAX_NUMBER_OF_FADC];

volatile int intr_count = 0;
volatile int close_err = 0;

int channel;
LW *buf = NULL;
int rc;
static int header2_check_mismatch =0;

static FILE *stream1,*stream2,*stream3;
/*static int thresh_read[NUMBER_OF_CHANNEL*MAX_NUMBER_OF_FADC];*/
static int thresh_read[TOTAL_NUM_CH];
static LW *block_start_addr[MAX_NUMBER_OF_FADC+1];
static int update_thresh = 0, update_irqevt = 1, number_of_words = 0;
static int sp8f_module, sp8f_channel, sp8f_event, sp8f_suppr;
static int fadc_data, data_id, data_value, first_fadc_number;
static int NUMBER_OF_FADC = 0;
static W module_flag, channel_flag, event_flag;
static int header_sum = 0, header1_sum = 0, header2_sum = 0, trailer_sum = 0; 
static int adc_counter = 0, tdc_counter = 0;
static unsigned int fadc_board_mask[2];
static int REG_FADC_VMEADR[MAX_NUMBER_OF_FADC];
static int REG_FADC_MODNUM[MAX_NUMBER_OF_FADC];
static int getFileName(char* optionFile, char* errFile);
static enum FADC_type fadc_type = NINE_U;      /*5407*/
static int Num_ch; /* the number of channel in one FADC   5407 */
static int ChOffset = NUMBER_OF_CHANNEL_9U*NUMBER_OF_FADC_9U -1; 
/*channel# offset for 6U = the last ch# for 9U  5407*/
static int error_flag=0;

enum data_type{BLOCK_END = -1,TRAILER,HEADER1,ADC,TDC,HEADER2,CHANNEL_CHANGE};

static int id ;
static caddr_t vaddr;

volatile int intr_flag;
volatile int intr_VMEread_flag;

int word_error=0;

/* function declaration */

#define data_read 1              /* Default = 1 */
#define data_check 1             /* Default = 1 */
#define VME_BLTREAD 1            /* Default = 1 */
#define PRINT_STRUCT 0           /* Default = 0 */

int num_IRQ;
void clean_error_file(char*);    /* Clean the error_log.dat                   */

#ifdef RAWDATA_PRINT
void clean_rawdata_file();    /* Clean the rawdata_log.dat                   */
#endif
#ifdef ERROR_WITH_RAW_PRINT
void clean_error_rawdata_file();    /* Clean the rawdata_log.dat                   */
#endif

void read_thresh();         /* Read the options of file SP8F_thresh.dat  */
void read_option(char*);         /* Read the options of file SP8F_option.dat  */
void check_sum();           /* Check check-sum bit of the fadc_data      */
void hamming_code_check();  /* Check Hamming bits of the fadc_data       */
int  identify();            /* Check the data type of the fadc_data      */
void display_header_hex();  /* Display the data in the HEX form          */
void check_adc_bin_sum();   /* Check counter value                       */
void check_suppr(int);      /*  Check the consistent of the supression   */
int  initialization();      /* Initialize the system                     */
void close_main();          /* Close all declaration                     */
void clean_identify_counter();
void read_mod_cha_eve_sup();
void check_mod_cha_eve_continue();
/* inline void reset_and_wait(); */
void reset_and_wait();
void event_structure_check(int);
void check_h1_h2_trail_sum();
void fadc_board_mask_extract();
int DATA_CHECK_ONLINE();

/* ========================================= */
/* Read FADC FIFO data to a memory block(buf)*/
/* pdata -> the start address of buf         */
/* each read -> size of the number_of_words  */
/* rc -> the total bytes sent to PC          */
/* ========================================= */

void sighdl( arg )                     /* Interupt handler    */
{
  struct timeval start, stop, echodelay;  /* start, stop and echo delay times */

  int i,ii, channel;
  LW *pdata = NULL;
  LW *pdata_begin = NULL;
  off_t blt_addr;
  int tmprc = 0;
  static int ifirst = 0;
  static int inactive[18];
  static int iloop[18];
  char* data_mark;
  int cADC,cTDC;
  LW *pdataloop = NULL;
  if (ifirst == 0) {
    ifirst = 1;
    for (i=0; i<18; i++) inactive[i]=0;
    for (i=0; i<18; i++) iloop[i]=0;
  }

  if((gettimeofday(&start, NULL)) == -1)  {
    perror("gettimeofday");
    exit(1);
  }

  /* disable interrupt and increment the counter */
  vui_intr_dis(fd, &intr);

  intr_count++;
  intr_flag = 1;
  intr_VMEread_flag = 0;
   
  if (data_read != 0)  {
    /* use local buffer pointer */
    pdata=buf - 1;
    block_start_addr[0] = buf;
	
    /* Read the FIFO */
    /*printf("check0 pdata = %d number_of_words %d\n",pdata,number_of_words);*/
    
    for(i=0; i<NUMBER_OF_FADC; i++) {
      
      if (VME_BLTREAD !=0)  {
	ii = i ;
	iloop[i] = 0;
	
	tmprc = 0;
	blt_addr = BLT_ADDR + REG_FADC_VMEADR[i]*BASE_OFFSET;
	
	do{
	  pdata++;
	  pdata_begin = pdata;
	  
	  iloop[i]++;
	  
	  if (lseek(fd1, blt_addr, SEEK_SET) == -1)
	    perror("read lseek");		
	  /*  rc:how many words has been transferred */
	  /*  printf("Read FADC %d blt_addr %x fd1 %x \n",i,blt_addr,fd1);*/

	  rc = read (fd1, (void *)pdata, sizeof(LW)*number_of_words);
	  
	  if (rc <=0) {
	    perror("read error");
	    printf("i= %d rc %d\n",i,rc);
	  }
	
#ifdef RAWDATA_PRINT
	  /*print every pdata*/
	  if (intr_count <11){
	    for (pdataloop=pdata;pdataloop< pdata + (number_of_words - 1);pdataloop++){
	      int ipdata; 
	      
	      if ((*pdataloop << 16) != (0xffff<<16)){
		fprintf(stream2,"%x ",*pdataloop);
		if (ipdata%10 == 0){
		  fprintf(stream2,"\n");		
  		}		

	      }else{
		fprintf(stream2,",");
	      }

	      ipdata++;
	    }  
	  }
	  /*------------------------------------------------------------------------------*/
#endif
	  pdata = pdata + (number_of_words - 1);
	  rc = rc/4;
	  tmprc = tmprc + rc;
	  
	  /*	  printf("Read FADC %d iloop %d pdata %x \n",i,iloop[i],*pdata);*/

	  /* 	} while (*pdata != BLOCK_END_IDENTIFY); */
	} while ((*pdata<<16) != 0xFFFF<<16);
	
#ifdef RAWDATA_PRINT	
	/*fflush(stream2);*/	
#endif
	
	/* new criteria for EMPTY FIFO*/
	if (iloop[i] <= 1 && (*pdata_begin<<16) == (BLOCK_END_IDENTIFY<<16) ) { 
	  inactive[i]++;
	} 
	
	block_start_addr[ii+1] = pdata+1;
	
      } else { /* VME_BLTREAD =0 */
	
	iloop[i]=0;
	
	/*	for (channel = 0; channel< NUMBER_OF_CHANNEL; channel++) {*/
	for (channel = 0; channel< Num_ch; channel++) { /*5407*/
	  int isAboveThre=0;
	  int thre=500;
	  
	  tmprc = 0;
	  
	  cADC = 0;
	  cTDC = 0;
	  
	  do {
	    printf("*pdata%x",pdata);
	    *++pdata = tpc[i]->CH[channel].DATA;
	    printf("*pdata%x",pdata);
	    iloop[i]++;

	    if (tmprc == 0)
	      pdata_begin = pdata;
	    tmprc++;
	    
	    /*************************/
	    data_id     = *pdata & ID_MASK;
	    data_value  = *pdata & VALUE_MASK;
	    
	    if (data_id == 0x0000) {
	      data_mark = "HED";
	    } else if (data_id == 0x2000) {
	      data_mark = "ADC";
	      cADC ++;
	      if ( data_value > thre ){
		isAboveThre = 1;
	      }
	      
	    }    else if (data_id == 0x4000) {
	      data_mark = "TDC";
	      cTDC ++;
	    } else if (data_id == 0x6000){
	      data_mark = "TRL";
	      if ( isAboveThre == 0){
		printf("ch %d is below threshold\n");
	      }
	    }
	    
	    if (*pdata<<16 == 0xffff<<16){
	      data_mark = "END";
	    }
	    
	    if (i==0&&channel==0)   {
	      printf ("IRQ %d FADC %d Ch %2d Bin %4d %s Data %d\n",
		      intr_count,i,channel,tmprc,data_mark,(data_value));
	    }
	    
	    if (cTDC >0)  {
	      cTDC = 0;
	      cADC = 0;
	    }
	    
	  } while(*pdata<<16 != 0xFFFF<<16);
	  if (tmprc == 1)
	    inactive[i]++;
	  
	  /*	  if (channel<(NUMBER_OF_CHANNEL-1)) */
	  if (channel<(Num_ch-1))  /*5407*/
	    pdata=pdata-1;
	}
	block_start_addr[i+1] = pdata+1;
	
      } /* BLT_READ */
    } /* NUMBER_OF_FADC */
    
    if (data_check != 0)
      DATA_CHECK_ONLINE();
    
    if((gettimeofday(&stop, NULL)) == -1)  {
      perror("gettimeofday");
      exit(1);
    }
    
    /* compute time delay */
    timeval_subtract(&echodelay, &stop, &start);
    
  }
  
  /* This makrs the finishing of VME READ */
  intr_VMEread_flag = 1;
  
  /* enable interrupt for the next event */
  
  vui_intr_ena(fd, &intr);

  if(intr_count < num_IRQ) {
    reset_and_wait();
  }else   {
    close_err = 1;
  }

  if ( (intr_count%1000 == 0)||((intr_count<=1000)&&(intr_count%100 == 0)) || ((intr_count<=100)&&(intr_count%10 == 0)) ){

#ifdef RAWDATA_PRINT
    fprintf(stream2,"*** Issuing Reset %d !\n",intr_count);
#endif
    printf("*** Issuing Reset %d !\n",intr_count); 
    fprintf(stream1,"*** Issuing Reset %d !\n",intr_count);
    /*for(i=0; i < NUMBER_OF_FADC; i++){
      printf("iFadc:%02d Base:%02d Mod:%02d iloop:%d\n",
	i,REG_FADC_VMEADR[i],REG_FADC_MODNUM[i],iloop[i]);
	}*/
  }
#ifdef RAWDATA_PRINT
  fflush(stream2);
#endif
  fflush(stream1); 
}

/**
 * this function is for computing the time difference between timeval x and y
 * the result is stored in result
 */
int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }
  
  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;
  
  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

/***********************************************************************/

main(argc,argv) int argc; char **argv; {

  struct timeval start, stop, echodelay;  /* start, stop and echo delay times */
  time_t start2, stop2;
  double diff;
  int iii;		
  printf("size of LW %d,size of W %d ",sizeof(LW),sizeof(W));
  if(initialization()==-1) exit(0);

  if((gettimeofday(&start, NULL)) == -1)  {
    perror("gettimeofday");
    exit(1);
  }

  iii = 0;
  
  while(intr_count < num_IRQ) {
    sleep(0);
  }

  
  if((gettimeofday(&stop, NULL)) == -1)  {
    perror("gettimeofday");
    exit(1);
  }
  
  /* compute time delay */

  
  timeval_subtract(&echodelay, &stop, &start);
  
  /* printf("Echo delay is %ds and %dus\n", echodelay.tv_sec, echodelay.tv_usec); */

  stop2 = time(NULL);
  diff = difftime(start2, stop2);

  if(close_err==1){
    close_main();
    fclose(stream1); /*  Close the opened file */
    fclose(stream2);
  }

  return 0; 

}

int DATA_CHECK_ONLINE(){

  int block_counter, i,  last_id = TRAILER;
  unsigned int *buf_redirect = NULL;
  int word_error_tmp = 0;
  
  clean_identify_counter();

  for(block_counter=0;block_counter<NUMBER_OF_FADC;block_counter++)  {

    buf_redirect = block_start_addr[block_counter];
    module_flag  = REG_FADC_MODNUM[block_counter];
    channel_flag = 0;
    event_flag   = 1;

    i = 0;
    do {
	
      fadc_data   = *(buf_redirect+i);
      
      /* printf("%d-%d 0x%8x\n",block_counter,i,fadc_data);*/
      if ((fadc_data<<16  == (0xffff<<16))&&(i==0)){
	printf("\n======== ERROR: empty FIFO : block counter %d Mod# %d IRQ %d ==========\n \n",
	       block_counter,module_flag,intr_count);
	fprintf(stream1,"ERROR: empty FIFO : block counter %d mod# %d IRQ %d\n",block_counter,
		module_flag, intr_count);
#ifdef ERROR_WITH_RAW_PRINT
	error_flag=1;
#endif
	word_error_tmp++;
      }
	
      data_id     = fadc_data & ID_MASK;
      data_value  = fadc_data & VALUE_MASK;
      /*      printf("fadc_data %x \n",fadc_data);*/
      /*    if ((update_irqevt*Num_ch) <= trailer_sum){ */  /*20060614*/
#ifdef RAWDATA_PRINT
      /*fprintf(stream2,"---if---(update_irqevt*Num_ch) %d,trailer_sum %d,fadc_data %x\n",(update_irqevt*Num_ch),trailer_sum,fadc_data);*/
#endif
																	/*}
      else{
      */
      id = identify();

      if(id == HEADER1)  {

	if(PRINT_STRUCT) printf("FADC %d HEADER1 %x  ",block_counter, data_value);	    
	read_mod_cha_eve_sup();

      }

      if(id == HEADER2) {
	if(PRINT_STRUCT) printf("HEADER2 %x  ",data_value);
	read_mod_cha_eve_sup();
	check_suppr(block_counter);
	check_mod_cha_eve_continue();
        if (header2_check_mismatch == 1){
	  printf("ERROR-Header1-information--ch_flag %d Mod %d sp8f_ch %d Evt %d Header1 %x IRQ %d\n",
                  channel_flag,sp8f_module,sp8f_channel,sp8f_event,*(buf_redirect+i-1),intr_count);
          fprintf(stream1,"ERROR-Header1-information--ch_flag %d Mod %d sp8f_ch %d Evt %d Header1 %x IRQ %d\n",
                  channel_flag,sp8f_module,sp8f_channel,sp8f_event,*(buf_redirect+i-1),intr_count);
#ifdef ERROR_WITH_RAW_PRINT
          fprintf(stream3,"ERROR-Header1-information--ch_flag %d Mod %d sp8f_ch %d Evt %d Header1 %x IRQ %d\n",
                  channel_flag,sp8f_module,sp8f_channel,sp8f_event,*(buf_redirect+i-1),intr_count);
	  error_flag=1;
#endif
	  
        }
	header2_check_mismatch =0;

      }
	    
      if(id == ADC) {
	hamming_code_check();
      }

      if(id == TDC) {
	hamming_code_check();
      }
      
      if(id == TRAILER) {
	if(PRINT_STRUCT)
	  printf("TRAILER %x \n",data_value);

	hamming_code_check();
	check_adc_bin_sum();

	if (adc_counter > 1018){
	  printf("ERROR: # of ADC %d : Mod %d Ch %d Evt %d\n",
		 adc_counter,sp8f_module,sp8f_channel,sp8f_event);
	  fprintf(stream1,"ERROR: # of ADC %d : Mod %d Ch %d Evt %d IRQ %d\n",
		  adc_counter,sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
	  fprintf(stream3,"ERROR: # of ADC %d : Mod %d Ch %d Evt %d IRQ %d\n",
		  adc_counter,sp8f_module,sp8f_channel,sp8f_event,intr_count);
	  error_flag=1;
#endif
	}
	adc_counter = 0;
      }
	    
      if(id == BLOCK_END){
	if(PRINT_STRUCT) printf("BLOCK_END %x \n",data_value);
      }

      event_structure_check(last_id);
      check_sum();
      last_id = id;

      /*       }*//*if ...else--20060614*/	
      i++;
#ifdef RAWDATA_PRINT
	/*	fprintf(stream2,"(update_irqevt*Num_ch) %d,trailer_sum %d,fadc_data %x\n",(update_irqevt*Num_ch),trailer_sum,fadc_data);*/
#endif
    /***************************************************************************/
      /*      } while (fadc_data != BLOCK_END_IDENTIFY);*/
      } while (fadc_data<<16 != BLOCK_END_IDENTIFY<<16);
    /***************************************************************************/ 

    check_h1_h2_trail_sum();
    clean_identify_counter();

#ifdef ERROR_WITH_RAW_PRINT
    if (error_flag==1){
      for(block_counter=0;block_counter<NUMBER_OF_FADC;block_counter++)  {
	
	buf_redirect = block_start_addr[block_counter];
	module_flag  = REG_FADC_MODNUM[block_counter];
	i=0;
	fprintf(stream3,"\n============== IRQ:%d ===Mod:%d=============\n\n",intr_count,module_flag);
	do {
	  
	  fadc_data   = *(buf_redirect+i);
	  fprintf(stream3,"%4x ",fadc_data&0xFFFF );
	  
	  i++;
	  if (i%10 == 0){
	    fprintf(stream3,"\n");                
	  } 
	  
	  
	} while (fadc_data<<16 != BLOCK_END_IDENTIFY<<16);
	fprintf(stream3,"\n===========================================\n");
      }
      error_flag=0;
    }
#endif


  } /* for data decoding */  
  if (word_error_tmp>0) word_error++;
  word_error_tmp = 0;
  
  return 0;
}

/* Clean the data inside the error_log.dat */
void clean_error_file(char* errFile) {
  int file_pointer;
  file_pointer = open(errFile,O_RDWR|O_TRUNC);
  close(file_pointer);
}
#ifdef RAWDATA_PRINT
/* Clean the data inside the rawdata_log.dat */
void clean_rawdata_file()
{
  int file_pointer;
  file_pointer = open(RAWDATA_FILE,O_RDWR|O_TRUNC);
  close(file_pointer);
}
#endif

#ifdef ERROR_WITH_RAW_PRINT
/* Clean the data inside the rawdata_log.dat */
void clean_error_rawdata_file()
{
  int file_pointer;
  file_pointer = open(ERROR_WITH_RAW_FILE,O_RDWR|O_TRUNC);
  close(file_pointer);
}
#endif




/* Read the suppressive level on each channel */
void read_thresh() {
  FILE *fp;
  int i;
  int xchan;
  int istatus,data,addr;
  fp = fopen(PEDESTAL_FILE,"r");
  if(fp == NULL){
    printf("Threshold file %s cannot be opened! \n",PEDESTAL_FILE);
    exit(8);
  }

  for( i=0; i<  NUMBER_OF_FADC*MAX_NUMBER_OF_FADC; i++)
    thresh_read[i] = 0;

  while (1){
    istatus = fscanf(fp,"%d %d \n",&xchan, &data);
    if (istatus<=0) break;
    addr = xchan;
/*    if (addr <= NUMBER_OF_CHANNEL*MAX_NUMBER_OF_FADC){*/
    if ( addr <= TOTAL_NUM_CH ){     /*5407*/
      thresh_read[addr] = data;
      
      if(DISPLAY_THRESH){
	fprintf(stream1,"PEDESTAL_FILE: %d addr %d \n",thresh_read[addr],addr);
      }
    }
  }
  fclose(fp);
}

/* Read option.dat to setup variable update_thresh & update_option */
void read_option(char* optionFile){
  FILE *fro;
  int i;
  int num_FADC_installed;
  int index_VME, index_MOD, index_enable;
	
  printf("Option file %s. \n",optionFile);
  
  fro=fopen(optionFile,"r");     
  if (fro == NULL){
    printf("Option file %s cannot be opened! \n",optionFile);
    exit(8);
  }
  NUMBER_OF_FADC = 0;
  fscanf(fro,"%d \n",&num_IRQ);
  fscanf(fro,"%d \n",&update_thresh);
  fscanf(fro,"%d \n",&update_irqevt);
  fscanf(fro,"%d \n",&number_of_words);
  /*
  fscanf(fro,"%x \n",&number_of_words);
  */
  fscanf(fro,"%d \n",&num_FADC_installed);

  printf("num_IRQ %d,update_thresh %d,update_irqevt %d,number_of_words %d \n",num_IRQ,update_thresh,update_irqevt,number_of_words);

  for (i=0; i<num_FADC_installed; i++) {
    fscanf(fro,"%d %d %d \n",&index_VME, &index_MOD, &index_enable);
    if (index_enable == 1){
      REG_FADC_VMEADR[NUMBER_OF_FADC] = (index_VME-1);
      REG_FADC_MODNUM[NUMBER_OF_FADC] = index_MOD;
      NUMBER_OF_FADC++;
      printf("FADC %d VME %d Mod %d Registered. \n",NUMBER_OF_FADC,
	     REG_FADC_VMEADR[NUMBER_OF_FADC-1],REG_FADC_MODNUM[NUMBER_OF_FADC-1]);
      fprintf(stream1,"FADC %d VME %d Mod %d Registered. \n",NUMBER_OF_FADC,
	      REG_FADC_VMEADR[NUMBER_OF_FADC-1],REG_FADC_MODNUM[NUMBER_OF_FADC-1]);
    }
  }
  printf("NUMBER_OF_FADC Registered = %d. \n",NUMBER_OF_FADC);

#ifdef DISPLAY_OPTION  
  printf("optionFile: thresh %d irqevt %d words %d fadc_board_mask %x %x \n", 
	 update_thresh, update_irqevt,number_of_words,fadc_board_mask[1],fadc_board_mask[0]);
  fprintf(stream1,"optionFile: thresh %d irqevt %d words %x fadc_board_mask %x %x \n", 
	  update_thresh, update_irqevt,number_of_words,fadc_board_mask[1],fadc_board_mask[0]);    
#endif

  fclose(fro);
}

/* check up of checksum */
/* ====================================================== */
/* Header1,2:       D15 = D0(XOR)D1(XOR)D2 .... (XOR)D10  */
/* ADC,TDC,Trailer: D15 = D0(XOR)D1(XOR)D2 .... (XOR)D9   */
/* ====================================================== */
void check_sum(){
  unsigned char iloop = 0;
  int ibit_checksum = 0;
  switch(id){
    /*  Header1,Header2 D0...D10 <-- 11 bits  */
  case HEADER1: 
    for (iloop=0; iloop<11; iloop++)
      ibit_checksum = ibit_checksum^(fadc_data>>iloop & 1);
    if(ibit_checksum != ((fadc_data>>15)&1)){
      printf("ERROR: Check Sum - H1  : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Sum - H1  : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Sum - H1  : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
      error_flag=1;
#endif
    }

    break;
  case HEADER2: 
    for (iloop=0; iloop<11; iloop++)
      ibit_checksum = ibit_checksum^(fadc_data>>iloop & 1);

    if(ibit_checksum != ((fadc_data>>15)&1)){
      printf("ERROR: Check Sum - H2  : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Sum - H2  : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Sum - H2  : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
      error_flag=1;
#endif
      
    }
    break;

    /*  ADC,TDC,Trailer D0,D1,D2....D9 <--- 10 bits */
  case ADC: 
    for (iloop=0; iloop<10; iloop++)
      ibit_checksum = ibit_checksum^(fadc_data>>iloop & 1);
    if(ibit_checksum != ((fadc_data>>15)&1)){
      printf("ERROR: Check Sum - ADC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Sum - ADC : Mod %d Ch %d Evt %d data %x IRQ %d \n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Sum - ADC : Mod %d Ch %d Evt %d data %x IRQ %d \n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      error_flag=1;
#endif
	
    }
    break;

  case TDC: 
    for (iloop=0; iloop<10; iloop++)
      ibit_checksum = ibit_checksum^(fadc_data>>iloop & 1);
    if(ibit_checksum != ((fadc_data>>15)&1)){
      printf("ERROR: Check Sum - TDC : Mod %d Ch %d Evt %d data %x  IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Sum - TDC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Sum - TDC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      error_flag=1;
#endif


    }
    break;

  case TRAILER:
    for (iloop=0; iloop<10; iloop++){
      ibit_checksum = ibit_checksum^(fadc_data>>iloop & 1);
    }
    if(ibit_checksum != ((fadc_data>>15)&1)){
      printf("ERROR: Check Sum - Tr : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Sum - Tr : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Sum - Tr : Mod %d Ch %d Evt %d IRQ %d data %x %d %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,intr_count,fadc_data,ibit_checksum,
	      (fadc_data>>15)&1);
      error_flag=1;
#endif
	
    }
    break;

  default:
    break;
  } /*  switch(id) end */

} 

/* Hamming code check                            */
/* ============================================= */
/* hamming_code1 = D12 = D15 (XOR) D14 (XOR) D13 */
/* hamming_code2 = D11 = D15 (XOR) D14 (XOR) D0  */
/* hamming_code3 = D10 = D15 (XOR) D13 (XOR) D0  */
/* ============================================= */
void hamming_code_check()
{
  unsigned int h1,h2,h3,d15,d14,d13,d12,d11,d10,d0;
  d15 = (fadc_data>>15)&1;
  d14 = (fadc_data>>14)&1;
  d13 = (fadc_data>>13)&1;
  d12 = (fadc_data>>12)&1;
  d11 = (fadc_data>>11)&1;
  d10 = (fadc_data>>10)&1;
  d0  =  fadc_data&1;
  h1 = d15^d14^d13;
  h2 = d15^d14^d0 ;
  h3 = d15^d13^d0 ;
  if((d12!=h1)||(d11!=h2)||(d10!=h3)){
    switch(id){
    case ADC: 
      printf("ERROR: Check Hamming - ADC : Mod %d Ch %d Evt %d data %x  IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Hamming - ADC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Hamming - ADC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      error_flag=1;
#endif
      break;
    case TDC: 
      printf("ERROR: Check Hamming - TDC; Module %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Hamming - TDC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Hamming - TDC : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      error_flag=1;
#endif	      
      break;
    case TRAILER: 
      printf("ERROR: Check Hamming - Tr  : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      fprintf(stream1,"ERROR: Check Hamming - Tr  : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: Check Hamming - Tr  : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
      error_flag=1;
#endif
      break;
    default: 
      break;
    } /*  switch(id){ end */
  } /* if((d12!=h1)||(d11!=h2)||(d10!=h3)){ end */
}

/* Check header type and calculation                   */
/* =================================================== */
/* Check the data structure of the data in memory      */
/* 0x0000 => 0000 XXXX XXXX XXXX ==> Header Data       */
/* 0x2000 => 0010 XXXX XXXX XXXX ==> ADC Data          */
/* 0x4000 => 0100 XXXX XXXX XXXX ==> TDC Data          */
/* 0x6000 => 0110 XXXX XXXX XXXX ==> Trail Data        */
/* =================================================== */
int identify() {
  if (SHOW_DATA_VALUE){
    printf("Orginal--%x ,Header--%x ,Data--%x \n",fadc_data,data_id,data_value);
    fprintf(stream1,"Orginal--%x ,Header--%x ,Data--%x \n",fadc_data,data_id,data_value);
  }

  switch(data_id){     
  case 0x0000: /*  Header data  */
    header_sum++;    
    /*Check HEADER1 or HEADER2 */
    if(header_sum%2 == 1){ /* Header1      */
      id = HEADER1;     
      header1_sum++;
      /* printf("fadc data: %x\n",fadc_data);*/
    }
    else{  /*  Header2 */
      id = HEADER2; 
      header2_sum++;
    } 
    /*
    printf("HEA Mod %d Ch %d Evt %d data %d IRQ %d\n",
	   sp8f_module,sp8f_channel,sp8f_event,(fadc_data & VALUE_MASK),intr_count);
    */
    break;
    
  case 0x2000: /*  ADC data */
    id = ADC;
    adc_counter++;
   
    /* if (sp8f_channel == 14) */
    {
      printf("ADC Mod %d Ch %d Evt %d data %d IRQ %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,(fadc_data & VALUE_MASK),intr_count);
    }

    break;
    
  case 0x4000: /*  TDC data */
    id = TDC;
    tdc_counter++;

    printf("TDC Mod %d Ch %d Evt %d data %d IRQ %d\n",
	   sp8f_module,sp8f_channel,sp8f_event,(fadc_data & VALUE_MASK),intr_count);
   

    break;
    
  case 0x6000: /*  Trailer data */
    if(fadc_data<<16 == BLOCK_END_IDENTIFY<<16){
      id = BLOCK_END;
      break;
    }
    id = TRAILER;
    trailer_sum++;
    /*
    printf("TRI Mod %d Ch %d Evt %d data %d IRQ %d\n",
	   sp8f_module,sp8f_channel,sp8f_event,(fadc_data & VALUE_MASK),intr_count);
    */
    break;    
    
  default: /*  impossible */
    printf("ERROR: id - Unknown : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	   sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
    fprintf(stream1,"ERROR: id - Unknown : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	    sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: id - Unknown : Mod %d Ch %d Evt %d data %x IRQ %d\n",
	    sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
    error_flag=1;
#endif
    break;
  }/* switch loop end  */


  return id;
}

/* Display the separated type header */
/* like this -->01 010 0101010 type  */
void display_header_hex(void)
{
  int iloop;
  if (data_id!=0x2000) {
    for (iloop=15; iloop>=15; iloop--){
      if (fadc_data & (1<<iloop))
	printf("1");
      else
	printf("0");
    }
    printf(" ");
    for (iloop=14; iloop>=13; iloop--){
      if (fadc_data & (1<<iloop))
	printf("1");
      else
	printf("0");
    }
    printf(" ");  
    for (iloop=12; iloop>=10; iloop--){
      if (fadc_data & (1<<iloop))
	printf("1");
      else
	printf("0");
    }
    printf(" ");
    for (iloop=9; iloop>=0; iloop--){
      if (fadc_data & (1<<iloop))
	printf("1");
      else
	printf("0");
    }
    printf("    %d ",fadc_data & VALUE_MASK);
    printf("\n");
  }
}

/* Check the adc bin summation */
void check_adc_bin_sum(void)
{
 
  if (adc_counter>0) 
    { 
      printf("Trailer summation: Tr %d sum %d Mod %d Ch %d Evt %d \n",data_value,adc_counter,sp8f_module,sp8f_channel,sp8f_event);
      if (adc_counter<4) 
	{
	  sleep(10);
	}
    }

  if(data_value != adc_counter){
    printf("ERROR: Tr sum imcompatible : Tr %d sum %d Mod %d Ch %d Evt %d IRQ %d\n",
	   data_value,adc_counter,sp8f_module,sp8f_channel,sp8f_event,intr_count);
    fprintf(stream1,"ERROR: Tr sum imcompatible : Tr %d sum %d Mod %d Ch %d Evt %d IRQ %d\n",
	    data_value,adc_counter,sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: Tr sum imcompatible : Tr %d sum %d Mod %d Ch %d Evt %d IRQ %d\n",
	    data_value,adc_counter,sp8f_module,sp8f_channel,sp8f_event,intr_count);
    error_flag=1;
#endif
  }
}

/* Check Suppressive level is equal or not to data in file */
void check_suppr(int block_counter)
{
  int ch;
 
  switch(fadc_type){
  case NINE_U:
    ch = (REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL_9U)+sp8f_channel;
    break;
  case SIX_U:
    ch = ChOffset + (REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL_6U)+sp8f_channel;
    break;
  }
 
  if(sp8f_suppr != thresh_read[ch]){

    printf("ERROR: Suppr-unequal :sp8f_suppr %d thresh_read %d Mod %d Ch %d Evt %d IRQ %d totalCh%d\n",
	   sp8f_suppr,
	   thresh_read[ch],
	   sp8f_module,sp8f_channel,sp8f_event,intr_count,
	   ch);
    fprintf(stream1,"ERROR: Suppr-unequal :sp8f_suppr %d thresh_read %d Mod %d Ch %d Evt %d IRQ %d\n",
	    sp8f_suppr,thresh_read[ch],
	    sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: Suppr-unequal :sp8f_suppr %d thresh_read %d Mod %d Ch %d Evt %d IRQ %d\n",
	    sp8f_suppr,thresh_read[ch],
	    sp8f_module,sp8f_channel,sp8f_event,intr_count);
    error_flag=1;
#endif
  }

/*
  if(sp8f_suppr != thresh_read[(REG_FADC_VMEADR[block_counter]
				*NUMBER_OF_CHANNEL)+sp8f_channel]){

    printf("ERR: Suppr-unequal :sp8f_suppr %d thresh_read %d Mod %d Ch %d Evt %d IRQ %d totalCh%d\n",
	   sp8f_suppr,
	   thresh_read[(REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL)+sp8f_channel],
	   sp8f_module,sp8f_channel,sp8f_event,intr_count,
	   (REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL)+sp8f_channel,
	   thresh_read[(REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL)+sp8f_channel+1]);
    fprintf(stream1,"ERR: Suppr-unequal :sp8f_suppr %d thresh_read %d Mod %d Ch %d Evt %d IRQ %d\n",
	    sp8f_suppr,thresh_read[(REG_FADC_VMEADR[block_counter]*NUMBER_OF_CHANNEL)+sp8f_channel],
	    sp8f_module,sp8f_channel,sp8f_event,intr_count);
  }
*/

}

/* Set the initial value and set device to ready */
int initialization()
{
  int win_size, win_base;
  bt_t win_flag;
  int i,status,ii,statgetFile;
  struct TPC_REG tpcReg;
  char optionFile[100], errFile[100];
  
/*get file name, hostname, and FADCtype  5407*/
  statgetFile = getFileName(optionFile, errFile);
  if(statgetFile==-1){
    printf("cannot get hostname\n");
  } else if(statgetFile==0) {
    printf("cannot find hostname in the list\n");
  }

  /*check FADCtype & set Num_ch   5407*/
  printf("FADC type : ");
  switch( fadc_type ){
  case NINE_U:
    printf("9U\n");
    Num_ch = NUMBER_OF_CHANNEL_9U;
    break;
  case SIX_U:
    printf("6U\n");
    Num_ch = NUMBER_OF_CHANNEL_6U;
    break;
  }
  
  /* clear the error_log.dat file */  
  clean_error_file(errFile); 
  stream1 = fopen(errFile,"a+"); 
  if (stream1 == NULL){ 
    printf("<ERROR> Write_open %s.\n",errFile);
    return -1;
  }
#ifdef RAWDATA_PRINT
  /* clear the rawdata_log.dat file */
  clean_rawdata_file(); 
  stream2 = fopen(RAWDATA_FILE,"a+");
  if (stream2 == NULL){ 
    printf("<ERROR> Write_open rawdata_log.dat.\n");
    return -1;
  }
#endif

#ifdef ERROR_WITH_RAW_PRINT
  /* clear the rawdata_log.dat file */
  clean_error_rawdata_file(); 
  stream3 = fopen(ERROR_WITH_RAW_FILE,"a+");
  if (stream3 == NULL){ 
    printf("<ERROR> Write_open error_rawdata_log.dat.\n");
    return -1;
  }
#endif

 
  read_option(optionFile);
  /*  fadc_board_mask_extract(); */

  module_flag  = REG_FADC_MODNUM[0];
  channel_flag = 0;
  event_flag   = 0;

  /* open the dma device */
  if ((fd1=open("/dev/vmedma24blt", O_RDWR)) == -1) { 
    perror("open vme24blt");
    return -1;
  }
  printf("VME device opened for BLT\n");
  
  /* open vmeplus for interrupt */
  if ((fd=open("/dev/vme24blt", O_RDWR)) == -1) {
    perror("open vme24blt");
    return -1;
  }
  printf("VME device opened for IRQ \n");
  
  /* mmap FADC address starting from 0x10000 */
  win_size = FADC_MAP_SIZE;
  win_base = BASE_ADDR;
  win_flag = PROT_READ | PROT_WRITE;
  
  /* ============================================= */
  /* write data to FADC FIFO with vme24blt mode(fd)*/
  /* directly map CSR(struct TPC_REG) to 0x10000  */
  /* ============================================= */
  vaddr = mmap((caddr_t)0, win_size, win_flag, MAP_SHARED, fd, win_base);
  printf("win_size %x win_flag %d MAP_SHARED %d\n",win_size,win_flag,MAP_SHARED);
  if (vaddr == (caddr_t)-1) {
    perror("Mapping");
    return -1;
  }
  
  /* allocate tpc structure */
  
  for (i=0; i<NUMBER_OF_FADC; i++) {
    tpc[i] = (struct TPC_REG *)(vaddr+REG_FADC_VMEADR[i]*BASE_OFFSET);
    printf("address of TPC_CSR is %p\n", &(tpc[i]->CSR));  
  }

  /*  tpc = (struct TPC_REG *)vaddr;  */
  printf("begin\n");

  /* allocate buffer for data storage */
  buf=(LW *)vui_dma_malloc(fd1,DMA_BUFFER_SIZE);
  if (buf == NULL) { perror("vui_dma_malloc"); exit(-1); }
  printf("buf = %p, end of buf = %p\n", buf, buf+0x7fffff);

  /* initial fadc and print out */
  /*  for (i=0; i<NUMBER_OF_FADC; i++) { */
  for (i=NUMBER_OF_FADC-1; i>=0; i--) {
    tpc[i]->CSR = TRIGGER_DISABLE;
    tpc[i]->CSR = RESET;
    tpc[i]->CSR = IRQ_EVENT_BASE + update_irqevt -1;
  }

  /* load pedestal supression level */
  if(update_thresh != 0){
    read_thresh();
    /*    for(i=0; i< NUMBER_OF_FADC; i++) { */
    for (i=NUMBER_OF_FADC-1; i>=0; i--) {
      /*      for(channel = 0; channel< NUMBER_OF_CHANNEL; channel++) {*/
      switch( fadc_type ){ /*5407*/
      case NINE_U:
	for(channel = NUMBER_OF_CHANNEL_9U-1; channel >=0; channel--) {
	  tpc[i]->CH[channel].DATA = 
	    thresh_read[REG_FADC_VMEADR[i]*NUMBER_OF_CHANNEL_9U+channel];
	  fprintf(stream1,"thresh_read %d\n",thresh_read[REG_FADC_VMEADR[i]*NUMBER_OF_CHANNEL_9U+channel]);
	}
	break;
      case SIX_U:
	for(channel = NUMBER_OF_CHANNEL_6U-1; channel >=0; channel--) {
	  tpc[i]->CH[channel].DATA = 
	    thresh_read[ChOffset + REG_FADC_VMEADR[i]*NUMBER_OF_CHANNEL_6U+channel];
	  fprintf(stream1,"thresh_read %d\n",
		  thresh_read[ChOffset + REG_FADC_VMEADR[i]*NUMBER_OF_CHANNEL_6U+channel]);
	}
      }      
    }
   
  }/*else if(update_thresh == 0){
    printf("update_thresh == 0\n");
    for(i=0; i< NUMBER_OF_FADC; i++) {
      for(channel = 0; channel< NUMBER_OF_CHANNEL; channel++) {
	thresh_read[REG_FADC_VMEADR[i]*NUMBER_OF_CHANNEL+channel]=4;
      }
      }
  }
    */
  
  /* set signal handler and enable interrupt */
  intr.level = 5;
  intr.prop = 0;
  intr.sig = SIGINT;
  
  /* Enable the direct mapping */
  sigset(SIGINT, sighdl);               /* Set signal handler */
  status = vui_intr_ena(fd, &intr);     /* enable interrupt   */

  printf("status vui_intr_ena : %d\n", status);
  if( status != VUI_OK ) {
    perror("vui_intr_ena");
    return -1;
  }

  printf("status vui_intr_ena --------------------------------------------------1\n");

  /* Enable trigger */
  for (i=0; i<NUMBER_OF_FADC; i++) { 
    /* the order is important here */
    tpc[NUMBER_OF_FADC-i-1]->CSR = TRIGGER_ENABLE;
  }
  printf("status vui_intr_ena --------------------------------------------------3\n");

  printf("trigger enabled, wait for triggers \n");
  return 0;
}

/* Disable and release all memory and functions*/
void close_main(){
  int i;
  for (i=0; i<NUMBER_OF_FADC; i++) {
    tpc[i]->CSR = TRIGGER_DISABLE;
  }
  vui_intr_dis(fd, &intr);
  sigset(SIGINT, SIG_DFL);
  
  free(buf);
  munmap(vaddr, FADC_MAP_SIZE);

  close(fd); close(fd1);
  printf("end \n");

  /* fclose(stream1);*/ /*  Close the opened file */

  printf("exit \n");
  
}

/* Clear all counter inside identify() */
void clean_identify_counter()
{
  /* Clear variable to 0 means one data structure end */
  tdc_counter = 0;
  header_sum  = 0;
  header1_sum = 0;
  header2_sum = 0;
  trailer_sum = 0;
}

/* Read module,channel,event,suppression value from Header1,Header2  */
/* ================================================================= */
/* Header1:XXXX XXXX XMMM MMMX XXXX =>  M: Module value  X:Don't care*/
/* Header1:XXXX XXXX XXXX XXXC CCCC =>  C: Channel value X:Don't care*/
/* ================================================================= */
/* Header2:XXXX XXXX XXXX XXXX XEEE =>  E: Event value   X:Don't care*/
/* Header2:XXXX XXXX XSSS SSSS SXXX =>  S: Suppression   X:Don't care*/ 
/* ================================================================= */
void read_mod_cha_eve_sup()
{
  if(id == HEADER1){ /*  Header1 */
    sp8f_module = (unsigned char)
      ((fadc_data & 0x000007E0) >> 5); /*  6 bits = 64 modules */
    sp8f_channel = (unsigned char)
      (fadc_data & 0x0000001F);        /*  5 bits = 32 channels */
  }
  if(id == HEADER2){ /* Header2 */
    sp8f_event = (unsigned char)
      (fadc_data & 0x00000007);        /* 3 bits = 8 events    */
    sp8f_suppr = (unsigned char)
      ((fadc_data & 0x000007F8) >> 3); /*  8 bits = 256 suppression levels */
  }
}

/* ================================= */
/* Check the module,channel,event is */
/* added by 1 on each measurement    */
/* ================================= */
void check_mod_cha_eve_continue()
{

  /* module => next */
    
  switch(sp8f_module - module_flag)
    {

    case 0: 
      
      switch(sp8f_channel - channel_flag){
      case 1: /*  channel => next  */
	/* event => 1 */
	/*	if(sp8f_event != 1){ */

	printf("ERROR: ch # mismtach : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x  IRQ %d\n",
	       channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	fprintf(stream1,"ERROR: ch # mismatch : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x IRQ %d\n",
		channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count); 
#ifdef ERROR_WITH_RAW_PRINT
	fprintf(stream3,"ERROR: ch # mismatch : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x IRQ %d\n",
		channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count); 
	error_flag=1;
#endif
	/* } */
	break;
	
      case 0:/*  channel => same  */
	/* event => next */

	/*	if((sp8f_event - event_flag) == 1){ */
	if((sp8f_event - event_flag) == 0){
	  break;
	}
	/* event => others */
	else{
	  printf("ERROR: eve # mismatch : eve_flag %d Mod %d Ch %d Evt %d data %x  IRQ %d\n",
		 event_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	  fprintf(stream1,"ERROR: eve # mismatch : eve_flag %d Mod %d Ch %d Evt %d data %x IRQ %d\n",
		  event_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
	  fprintf(stream3,"ERROR: eve # mismatch : eve_flag %d Mod %d Ch %d Evt %d data %x IRQ %d\n",
		  event_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	  error_flag=1;
#endif
	}
	break;

      default:
	header2_check_mismatch=1; 

	printf("ERROR: ch # mismatch : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x IRQ %d\n",
	       channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	fprintf(stream1,"ERROR: ch # mismatch : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x IRQ %d\n",
		channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count); 
#ifdef ERROR_WITH_RAW_PRINT
	fprintf(stream3,"ERROR: ch # mismatch : ch_flag %d Mod %d sp8f_ch %d Evt %d data %x IRQ %d\n",
		channel_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count); 
	error_flag=1;
#endif
      }
      
      break;
      
    default: 
      if((sp8f_event != 1)||(sp8f_channel != 0)){
	printf("ERROR: mod # mismatch : mod_flag %d Mod %d Ch %d Evt %d data %x  IRQ %d\n",
	       module_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	fprintf(stream1,"ERROR: mod # mismatch : mod_flag %d Mod %d Ch %d Evt %d data %x IRQ %d\n",
		module_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
	fprintf(stream3,"ERROR: mod # mismatch : mod_flag %d Mod %d Ch %d Evt %d data %x IRQ %d\n",
		module_flag,sp8f_module,sp8f_channel,sp8f_event,fadc_data,intr_count);
	error_flag=1;
#endif
      }
      break;  
    }   
  
  if (event_flag < update_irqevt) 
    {
      event_flag++;
    }
  else
    {
      event_flag = 1;
      channel_flag++;
      /* printf ("Read Module %d -- finished \n !",sp8f_module); */
    }
  
  /*
    module_flag  = sp8f_module;
    channel_flag = sp8f_channel;
    event_flag   = sp8f_event;
  */

}

/* ================================================== */
/* Check structure complete using following algorithm */
/*      Header1 Header2 ADC TDC Trailer BLOCK_END     */
/* id =>   1       4     2   3     0        -1        */
/* Header1 -> if(id!=0) wrong                         */
/* Header2 -> if(id!=HEADER1) wrong                   */
/* ADC -> if(!((id == ADC)||(id == HEADER2))) wrong   */
/* TDC -> if(!((id == ADC)||(id == HEADER2))) wrong   */
/* Trailer -> if(id != TDC) wrong                     */    
/* BLOCK_END -> if(id != Trailer) wrong               */                     
/* ================================================== */
void event_structure_check(int last_id)
{
  switch(id){
  case HEADER1:
    if(!((last_id == TRAILER)||(last_id == BLOCK_END))){
      printf("ERROR: id - H1  : Mod %d Ch %d Evt %d data %x id %d lastId %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id);
      fprintf(stream1,"ERROR: id - H1  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data, id, last_id, intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - H1  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data, id, last_id, intr_count);
      error_flag=1;
#endif
    }
    break;
  case HEADER2:
    if(last_id != HEADER1){
      printf("ERROR: id - H2  : Mod %d Ch %d Evt %d data %x id %d lastId %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id);
      fprintf(stream1,"ERROR: id - H2  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - H2  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
      error_flag=1;
#endif
    }
    break;
  case ADC:
    if(!((last_id == ADC)||(last_id == TDC) ||(last_id == HEADER2))){
      printf("ERROR: id - ADC : Mod %d Ch %d Evt %d data %x id %d lastId %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id);
      fprintf(stream1,"ERROR: id - ADC : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - ADC : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
      error_flag=1;
#endif
    }
    break;
  case TDC:
    if(last_id != ADC){
      printf("ERROR: id - TDC : Mod %d Ch %d Evt %d data %x id %d lastid %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id );
      fprintf(stream1,"ERROR: id - TDC: Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - TDC: Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
      error_flag=1;
#endif
    }
    break;
  case TRAILER:
    if(!((last_id == TDC)||(last_id == HEADER2))){
      printf("ERROR: id - Tr  : Mod %d Ch %d Evt %d data %x id %d lastId %d\n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id);
      fprintf(stream1,"ERROR: id - Tr  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - Tr  : Mod %d Ch %d Evt %d data %x id %d lastId %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
      error_flag=1;
#endif
    }
    break;
  case BLOCK_END:
    if(last_id != TRAILER){
      printf("ERROR: id - BLOCK_END: Mod %d Ch %d Evt %d data %x id %d lastID %d \n",
	     sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id);
      fprintf(stream1,"ERROR: id - BLOCK_END: Mod %d Ch %d Evt %d data %x id %d lastID %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
      fprintf(stream3,"ERROR: id - BLOCK_END: Mod %d Ch %d Evt %d data %x id %d lastID %d IRQ %d\n",
	      sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,last_id,intr_count);
      error_flag=1;
#endif
    }
    break;

  default:
    printf("ERROR: id - Unknown: Mod %d Ch %d Evt %d data %x id %d\n",
	   sp8f_module,sp8f_channel,sp8f_event,fadc_data,id);
    fprintf(stream1,"ERROR: id - Unknown: Mod %d Ch %d Evt %d data %x id %d IRQ %d\n",
	    sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: id - Unknown: Mod %d Ch %d Evt %d data %x id %d IRQ %d\n",
	    sp8f_module,sp8f_channel,sp8f_event,fadc_data,id,intr_count);
    error_flag=1;
#endif
    break;
  }
}

/* inline void reset_and_wait() */
void reset_and_wait()
{
  int i;
  intr_flag = 0;
  for (i=NUMBER_OF_FADC-1; i>=1; i--){
    tpc[i]->CSR = RESET;
  }

  tpc[0]->CSR = RESET;
}


void check_h1_h2_trail_sum()
{

/*  int summation = update_irqevt*NUMBER_OF_CHANNEL;*/
  int summation = update_irqevt*Num_ch;

  if(header1_sum != summation){
    printf("ERROR: H1_sum : H1_sum %d summation %d  Mod %d Ch %d Evt %d\n",
	   header1_sum,summation,sp8f_module,sp8f_channel,sp8f_event);
    fprintf(stream1,"ERROR: H1_sum : H1_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    header1_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: H1_sum : H1_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    header1_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
    error_flag=1;
#endif
  }

  if(header2_sum != summation){
    printf("ERROR: H2_sum : H2_sum %d summation %d  Mod %d Ch %d Evt %d \n",
	   header2_sum,summation,sp8f_module,sp8f_channel,sp8f_event);
    fprintf(stream1,"ERROR: H2_sum : H2_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    header2_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: H2_sum : H2_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    header2_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
    error_flag=1;
#endif
  }

  if(trailer_sum != summation){
    printf("ERROR: Tr_sum : Tr_sum %d summation %d  Mod %d Ch%d Evt %d \n",
	   trailer_sum,summation,sp8f_module,sp8f_channel,sp8f_event);
    fprintf(stream1,"ERROR: Tr_sum : Tr_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    trailer_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
#ifdef ERROR_WITH_RAW_PRINT
    fprintf(stream3,"ERROR: Tr_sum : Tr_sum %d summation %d  Mod %d Ch %d Evt %d IRQ %d\n",
	    trailer_sum,summation,sp8f_module,sp8f_channel,sp8f_event,intr_count);
    error_flag=1;
#endif
  }

}

void fadc_board_mask_extract() /*not used function*/
{
  int i,j,bit_check;
  unsigned int extract_mask = 0x000000FF;
  fadc_board_mask[1] = fadc_board_mask[1] & extract_mask;
  for(j=0;j<2;j++){
    for(i=0;i<32;i++){
      bit_check = (fadc_board_mask[j]>>i)&1;
      if(bit_check == 1){
        REG_FADC_VMEADR[NUMBER_OF_FADC] = j*32+i;
	/*        REG_FADC_MODNUM[NUMBER_OF_FADC] = MODNUM[j*32+i]; */
	NUMBER_OF_FADC++;
      }
    }
  }
  printf("NUMBER_OF_FADC %d \n",NUMBER_OF_FADC);

}

static int getFileName(char* optionFile, char* errFile)
{
    char *hostname;

    hostname = (char*) getenv("HOST");
    printf("hostname %s",hostname);
    if (hostname == NULL) return -1; /* Cannot get the hostname. */

    if (strcmp(hostname, TPC0CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC0);
      sprintf(errFile,    ERR_FILE_TPC0);
      fadc_type = FADC_TYPE_TPC0;
      return 1;
    } else if (strcmp(hostname, TPC1CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC1);
      sprintf(errFile,    ERR_FILE_TPC1);
      fadc_type = FADC_TYPE_TPC1;
      return 1;
    } else if (strcmp(hostname, TPC2CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC2);
      sprintf(errFile,    ERR_FILE_TPC2);
      fadc_type = FADC_TYPE_TPC2;
      return 1;
    } else if (strcmp(hostname, TPC3CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC3);
      sprintf(errFile,    ERR_FILE_TPC3);
      fadc_type = FADC_TYPE_TPC3;
      return 1;
    } else if (strcmp(hostname, TPC4CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC4);
      sprintf(errFile,    ERR_FILE_TPC4);
      fadc_type = FADC_TYPE_TPC4;
      return 1;
    } else if (strcmp(hostname, TPC5CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC5);
      sprintf(errFile,    ERR_FILE_TPC5);
      fadc_type = FADC_TYPE_TPC5;
      return 1;
    } else if (strcmp(hostname, TPC6CPU) == 0) {
      sprintf(optionFile, OPTION_FILE_TPC6);
      sprintf(errFile,    ERR_FILE_TPC6);
      fadc_type = FADC_TYPE_TPC6;
      return 1;
    }     
    return 0; /* The hostname was not fount in the list. */
}

