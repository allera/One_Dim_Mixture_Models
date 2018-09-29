function save_mmfit(x,src,path)
%saves plots
%INPUT: x is a data vector;
%       src is output of SIN_VB_MixMod.m (and of init?)
%       path is the path to the future figure we will create
%OUTPUT: 
x_or=x;
str=path;
                h1= figure(1);clf
                FontSize=20;
                % some common properties
                set(h1,'PaperType','A4'); 
                set(h1,'PaperOrientation','portrait');
                set(h1,'PaperUnits','centimeters');
                set(h1,'PaperPosition',[0 0 20 20])
                plt=src;%{1}{1};%src{anyone}{i};
                [f,x]=hist(x,50);
                bar(x,f/trapz(x,f));
                hold on
                ax=get(gca);
                cat=ax.Children;
                set(cat(1),'FaceColor',[145 25 206]/255,'BarWidth',2);
                invgam=@(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
                rage=-10:.001:10;
                pos=find(rage>0);neg=find(rage<0);
                
                if src.opts.MM=='GIM'
                    plt1=invgam(rage,plt.shapes(1),plt.scales(1));plt1(neg)=0;
                    plt2=invgam(-rage, plt.shapes(2),plt.scales(2)   );plt2(pos)=0;
                    title('GIM fit','FontSize',FontSize)
                else
                    plt1=gampdf(rage,plt.shapes(1),1/plt.rates(1));plt1(neg)=0;
                    plt2=gampdf(-rage,plt.shapes(2),1/plt.rates(2));plt2(pos)=0;
                    title('GGM fit','FontSize',FontSize)
                end
                plot(rage,plt.pi(1).*normpdf(rage,plt.mu1,sqrt(1/plt.tau1)),'r');hold on
                plot(rage,plt.pi(2).*plt1,'r');hold on
                plot(rage,plt.pi(3).*plt2,'r');hold on
                axis tight
                if numel(x_or)>0
                    ylim=max(f/trapz(x,f));
                else
                    ylim=.5;
                end
                set(gca,'ylim',[0 ylim])
                set(gca,'xlim',[-10 10])
                saveas(h1,str, 'psc2')
end